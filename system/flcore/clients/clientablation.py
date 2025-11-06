import time
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clientbase import Client


class clientablation(Client):
    """
    단순 FedProto 클라이언트:
      - local prototype 생성 (클래스별 평균)
      - local CE 학습
      - global prototype alignment (proto pull loss)
    """
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = args
        self.id = id
        self.device = torch.device(args.device)

        self.num_classes = args.num_classes
        self.local_epochs = args.local_epochs
        self.lambda_proto_pull = 1.0

        self.criterion_ce = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()

        self.global_protos = {}
        self.local_protos = {}

    # -------------------------------------------------------
    # local training
    # -------------------------------------------------------
    def train(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.local_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                reps = self.model.base(x)
                logits = self.model.head(reps)
                ce_loss = self.criterion_ce(logits, y)

                # --- Proto Pull Loss ---
                pull_loss = torch.tensor(0.0, device=self.device)
                if self.global_protos:
                    matched, target_p = [], []
                    for i, label in enumerate(y.tolist()):
                        if label in self.global_protos:
                            matched.append(reps[i])
                            target_p.append(self.global_protos[label].to(self.device))
                    if matched:
                        matched = torch.stack(matched)
                        target_p = torch.stack(target_p)
                        pull_loss = F.mse_loss(matched, target_p)

                total_loss = ce_loss + self.lambda_proto_pull * pull_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        return

    # -------------------------------------------------------
    # prototype collection
    # -------------------------------------------------------
    def collect_protos(self):
        self.model.eval()
        trainloader = self.load_train_data()
        emb_list = defaultdict(list)
        with torch.no_grad():
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                reps = self.model.base(x)
                for i, label in enumerate(y):
                    emb_list[int(label)].append(reps[i])

        local_proto = {}
        for cls_id, embs in emb_list.items():
            if len(embs) == 0:
                continue
            embs = torch.stack(embs)
            local_proto[cls_id] = embs.mean(dim=0, keepdim=True).detach().cpu()

        self.local_protos = local_proto
        return {"shared": local_proto}

    # -------------------------------------------------------
    # receive global prototypes
    # -------------------------------------------------------
    def set_protos(self, global_shared):
        self.global_protos = global_shared
