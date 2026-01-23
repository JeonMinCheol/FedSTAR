import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flcore.servers.serverbase import Server
from flcore.clients.clienttgp import clientTGP 


class FedTGP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientTGP)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes

        self.server_learning_rate = args.local_learning_rate
        self.batch_size = args.batch_size
        self.server_epochs = args.server_epochs
        self.margin_threthold = args.margin_threthold

        self.feature_dim = args.feature_dim
        self.server_hidden_dim = self.feature_dim

        # ===== in-memory TGP / global protos =====
        self.TGP = Trainable_Global_Prototypes(
            self.num_classes,
            self.server_hidden_dim,
            self.feature_dim,
            self.device
        ).to(self.device)

        self.TGP_opt = torch.optim.SGD(self.TGP.parameters(), lr=self.server_learning_rate)

        self.global_protos = None  # dict[int] -> tensor[d]

        self.CEloss = nn.CrossEntropyLoss()

        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        self.min_gap = None
        self.max_gap = None

        print(self.TGP)

    def train(self):
        for r in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if r % self.eval_gap == 0:
                print(f"\n-------------Round number: {r}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate(r)

            # ★ 라운드 시작 시 global protos를 클라이언트에 주입 (in-memory)
            for c in self.selected_clients:
                if hasattr(c, "set_protos"):
                    c.set_protos(self.global_protos)

            max_workers = min(self.max_parallel_clients, len(self.selected_clients))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(c.train) for c in self.selected_clients]
                for f in as_completed(futures):
                    f.result()

            self.receive_protos()
            self.update_TGP()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")

    def receive_protos(self):
        assert len(self.selected_clients) > 0

        self.uploaded_ids = []
        self.uploaded_protos = []              # list of (proto_tensor[d], class_id_int)
        uploaded_protos_per_client = []        # list of dict[class]->proto

        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)

            # ★ 디스크 로드 제거: client.protos 직접 사용
            protos = getattr(client, "protos", None)
            if protos is None:
                continue

            for k, p in protos.items():
                self.uploaded_protos.append((p.detach(), int(k)))
            uploaded_protos_per_client.append(protos)

        # class-wise minimum distance 계산
        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        avg_protos = proto_cluster(uploaded_protos_per_client)

        # avg_protos 비었으면 갭 업데이트 스킵
        if len(avg_protos) == 0:
            self.min_gap = torch.tensor(0.0, device=self.device)
            self.max_gap = torch.tensor(0.0, device=self.device)
            return

        keys = list(avg_protos.keys())
        for i in range(len(keys)):
            for j in range(i):
                k1, k2 = keys[i], keys[j]
                dis = torch.norm(avg_protos[k1] - avg_protos[k2], p=2)
                self.gap[k1] = torch.min(self.gap[k1], dis)
                self.gap[k2] = torch.min(self.gap[k2], dis)

        self.min_gap = torch.min(self.gap)
        for i in range(len(self.gap)):
            if self.gap[i] > torch.tensor(1e8, device=self.device):
                self.gap[i] = self.min_gap
        self.max_gap = torch.max(self.gap)

        print('class-wise minimum distance', self.gap)
        print('min_gap', self.min_gap)
        print('max_gap', self.max_gap)

    def update_TGP(self):
        if len(self.uploaded_protos) == 0:
            # 업로드된 proto가 없으면 global_protos 갱신 못함
            return

        self.TGP.train()

        for e in range(self.server_epochs):
            proto_loader = DataLoader(self.uploaded_protos, batch_size=self.batch_size,
                                      drop_last=False, shuffle=True)

            for proto, y in proto_loader:
                # proto: [B, d] / y: [B]
                proto = proto.to(self.device)
                y = torch.as_tensor(y, dtype=torch.long, device=self.device)

                proto_gen = self.TGP(list(range(self.num_classes)))  # [K, d]

                # pairwise euclidean distance: sqrt(||x||^2 - 2x·c + ||c||^2)
                features_square = torch.sum(proto.pow(2), dim=1, keepdim=True)           # [B,1]
                centers_square = torch.sum(proto_gen.pow(2), dim=1, keepdim=True)       # [K,1]
                features_into_centers = proto @ proto_gen.T                             # [B,K]
                dist2 = features_square - 2 * features_into_centers + centers_square.T # [B,K]

                # 수치 안전장치
                dist2 = torch.clamp(dist2, min=0.0)
                dist = torch.sqrt(dist2 + 1e-12)

                one_hot = F.one_hot(y, self.num_classes).float()
                margin = min(self.max_gap.item(), self.margin_threthold) if self.max_gap is not None else self.margin_threthold
                dist = dist + one_hot * margin

                loss = self.CEloss(-dist, y)

                self.TGP_opt.zero_grad(set_to_none=True)
                loss.backward()
                self.TGP_opt.step()

        print(f'Server loss: {loss.item()}')
        self.uploaded_protos = []

        # ===== in-memory global protos 갱신 =====
        self.TGP.eval()
        global_protos = {}
        with torch.no_grad():
            for class_id in range(self.num_classes):
                global_protos[class_id] = self.TGP(class_id).detach()
        self.global_protos = global_protos


def proto_cluster(protos_list):
    proto_clusters = defaultdict(list)
    for protos in protos_list:
        for k, p in protos.items():
            proto_clusters[int(k)].append(p.detach())

    for k in list(proto_clusters.keys()):
        protos = torch.stack(proto_clusters[k], dim=0)
        proto_clusters[k] = torch.mean(protos, dim=0).detach()

    return proto_clusters


class Trainable_Global_Prototypes(nn.Module):
    def __init__(self, num_classes, server_hidden_dim, feature_dim, device):
        super().__init__()
        self.device = device
        self.embedings = nn.Embedding(num_classes, feature_dim)
        self.middle = nn.Sequential(
            nn.Linear(feature_dim, server_hidden_dim),
            nn.ReLU()
        )
        self.fc = nn.Linear(server_hidden_dim, feature_dim)

    def forward(self, class_id):
        # class_id: int | list[int] | tensor
        if not torch.is_tensor(class_id):
            class_id = torch.tensor(class_id, device=self.device)
        else:
            class_id = class_id.to(self.device)

        emb = self.embedings(class_id)
        mid = self.middle(emb)
        out = self.fc(mid)
        return out
