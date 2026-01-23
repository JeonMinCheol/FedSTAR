import time
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flcore.clients.clientbase import Client

# ----------------------------------------------------------------------------
# 1. 보조 모듈 (StyleFiLM, ClientPrototypeGenerator) - 그대로 유지
# ----------------------------------------------------------------------------
class StyleFiLM(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gamma = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh()
        )
        self.beta = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh()
        )

    def forward(self, x, style):
        gamma = self.gamma(style)
        beta = self.beta(style)
        return x * (1 + gamma) + beta

class ClientPrototypeGenerator(nn.Module):
    def __init__(self, model: nn.Module, num_classes: int):
        super().__init__()
        embed_dim = 0
        if isinstance(model.head, nn.Sequential):
            for layer in reversed(model.head):
                if isinstance(layer, nn.Linear):
                    embed_dim = layer.in_features
                    break
        elif isinstance(model.head, nn.Linear):
            embed_dim = model.head.in_features
        else:
            embed_dim = 576 
            
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(embed_dim)
        
        self.personal_table = nn.Parameter(torch.zeros(num_classes, embed_dim))
        nn.init.zeros_(self.personal_table)

    def forward(self, emb_list: dict) -> dict:
        device = self.personal_table.device
        protos = {}
        for cls_id, embs_per_class in emb_list.items():
            if len(embs_per_class) == 0: continue
            embs = torch.stack(embs_per_class).to(device)
            embs = self.norm(embs)
            shared = embs.mean(dim=0, keepdim=True)
            personal = self.personal_table[cls_id].unsqueeze(0)
            full = shared + personal
            protos[cls_id] = {"shared": shared, "personal": personal, "full": full}
        return protos

# ----------------------------------------------------------------------------
# 2. clientstar 메인 클래스 (수정됨)
# ----------------------------------------------------------------------------
class clientstar(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = args
        self.id = id
        self.device = torch.device(args.device)
        
        # 🎛️ [Toggle] FiLM 사용 여부 스위치 (기본값: True)
        # args에 use_film이 없으면 기본적으로 켜도록 설정 (원하시면 False로 변경 가능)
        self.use_film = self.args.use_film

        self.global_shared = {}
        self.global_protos = {}
        self.personalized_protos = {}
        self.local_personal_protos = {}

        self.num_classes = args.num_classes
        self.local_epochs = args.local_epochs

        self.criterion_ce = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()

        self.model.to(self.device)
        self.embed_dim = self.model.head.in_features

        # 모듈 초기화
        self.proto_gen = ClientPrototypeGenerator(self.model, num_classes=self.num_classes).to(self.device)
        if self.proto_gen.embed_dim == 0 or self.proto_gen.embed_dim != self.embed_dim:
             self.proto_gen.embed_dim = self.embed_dim
             self.proto_gen.norm = nn.LayerNorm(self.embed_dim).to(self.device)
             self.proto_gen.personal_table = nn.Parameter(torch.zeros(self.num_classes, self.embed_dim).to(self.device))

        # 🎛️ [Conditional Init] FiLM 모듈 조건부 생성
        if self.use_film:
            self.film = StyleFiLM(self.embed_dim).to(self.device)
        else:
            self.film = None

        # 🎛️ [Optimizer] FiLM 파라미터 조건부 등록
        optim_params = [
            {'params': self.model.parameters(), 'lr': self.learning_rate},
            {'params': self.proto_gen.parameters(), 'lr': self.learning_rate * 10}
        ]
        
        if self.use_film:
            optim_params.append({'params': self.film.parameters(), 'lr': self.learning_rate})
            
        self.optimizer = torch.optim.SGD(optim_params)

    def train(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()
        start_time = time.time()

        G_whole = torch.zeros(self.num_classes, self.embed_dim).to(self.device)
        if hasattr(self, 'global_protos') and self.global_protos:
            for label, proto in self.global_protos.items():
                if label < self.num_classes:
                    G_whole[label] = proto.to(self.device).squeeze()
            G_whole = F.normalize(G_whole, dim=1)

        for epoch in range(self.local_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                
                # SDM Logic
                lam = np.random.beta(1.0, 1.0)
                index = torch.randperm(x.size(0)).to(self.device)
                
                mixed_x = lam * x + (1 - lam) * x[index]
                y_a, y_b = y, y[index]
                
                if hasattr(self.model, 'base'): 
                    reps_raw = self.model.base(mixed_x)
                else: 
                    reps_raw = self.model(mixed_x)
                reps_raw = reps_raw.view(reps_raw.size(0), -1)
                
                # 🎛️ [Conditional Logic] FiLM 적용 여부 확인
                reps_personalized = reps_raw # 기본값 (Bypass)
                
                # 1. 스위치가 켜져 있고(use_film)
                # 2. 로컬 스타일 정보가 존재할 때만 실행
                if self.use_film and hasattr(self, "local_style") and self.local_style:
                    style_batch = []
                    # y_a 기준으로 스타일 생성 (간소화 버전)
                    for yy in y_a.tolist():
                        s = self.local_style.get(yy)
                        if isinstance(s, torch.Tensor): 
                            style_batch.append(s.squeeze(0).to(self.device))
                        else: 
                            style_batch.append(torch.zeros_like(reps_raw[0]))
                    
                    if style_batch:
                        style_batch = torch.stack(style_batch, dim=0)
                        reps_personalized = self.film(reps_raw, style_batch)

                # Classifier Head
                if hasattr(self.model, 'head'):
                    logits = self.model.head(reps_personalized)
                else:
                    logits = self.model.fc(reps_personalized)
                
                ce_loss = lam * self.criterion_ce(logits, y_a) + (1 - lam) * self.criterion_ce(logits, y_b)

                # Prototype Mixup Logic
                g_a = G_whole[y_a]
                g_b = G_whole[y_b]

                p_table = self.proto_gen.personal_table
                p_a = p_table[y_a]
                p_b = p_table[y_b]

                p_full_a = g_a + p_a
                p_full_b = g_b + p_b
                
                p_mixed = lam * p_full_a + (1 - lam) * p_full_b
                p_mixed = F.normalize(p_mixed, dim=1)
                
                z_norm = F.normalize(reps_personalized, dim=1)
                align_loss = (1 - (z_norm * p_mixed).sum(dim=1)).mean()

                total_loss = ce_loss # + align_loss 

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
        
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    # -------------------------------------------------------------------------
    # Server-Client Communication Logic (동일)
    # -------------------------------------------------------------------------
    def set_protos(self, global_shared):
        self.global_protos = global_shared
        self.personalized_protos = {}
        
        if not hasattr(self, "local_personal_protos") or not self.local_personal_protos:
            self.personalized_protos = copy.deepcopy(global_shared)
            return

        shared_labels = set(self.global_protos.keys()) & set(self.local_personal_protos.keys())
        for lbl in shared_labels:
            g = self.global_protos[lbl].to(self.device)
            p = self.local_personal_protos[lbl].to(self.device)
            personalized = g + p
            self.personalized_protos[lbl] = personalized.detach().cpu() 
            
        only_global = set(self.global_protos.keys()) - set(self.local_personal_protos.keys())
        for lbl in only_global:
            self.personalized_protos[lbl] = self.global_protos[lbl].detach().cpu()

    def collect_protos(self):
        self.model.eval()
        trainloader = self.load_train_data()
        emb_list = defaultdict(list)
        with torch.no_grad():
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                if hasattr(self.model, 'base'):
                    reps = self.model.base(x)
                else:
                    reps = self.model(x)
                reps = reps.view(reps.size(0), -1)
                for i, label in enumerate(y):
                    emb_list[int(label)].append(reps[i])
        
        local_proto_dict = self.proto_gen(emb_list)
        shared = {k: v["shared"] for k, v in local_proto_dict.items()}
        personal = {k: v["personal"] for k, v in local_proto_dict.items()}
        self.local_personal_protos = personal
        return {"shared": shared}
    
    def decompose_with_global(self, global_shared): # content & style 분해
        self.local_style = {}
        self.local_content = {}
        eps = 1e-6
        for lbl, local_p in self.local_personal_protos.items():
            if lbl not in global_shared: continue
            g = global_shared[lbl].to(self.device)
            p = local_p.to(self.device)
            proj = (p @ g.T) / (g @ g.T + eps) * g
            style = p - proj
            style = style / (style.norm(dim=-1, keepdim=True) + eps)
            self.local_content[lbl] = g.detach()
            self.local_style[lbl] = style.detach()
        return self.local_content, self.local_style

    def extract_features_for_retrieval(self, num_samples=20, mode="head", global_protos=None):
        """
        mode:
        - "head"  : g = normalize(head.weight[y])
        - "mean"  : g = normalize(mean_{x in class y} normalize(base(x)))   (local class mean feature)
        - "global": g = normalize(global_protos[y])  (server global proto)

        returns list of dict:
        {"client_id", "label", "content", "style", "cos_zg"}
        """
        self.model.eval()
        self.model.to(self.device)

        trainloader = self.load_train_data(batch_size=num_samples)

        extracted = []
        eps = 1e-12

        # ---------------------------
        # 1) collect (z, y)
        # ---------------------------
        Z_list, Y_list = [], []
        count = 0
        with torch.no_grad():
            for x, y in trainloader:
                if count >= num_samples:
                    break
                x = x.to(self.device)
                y = y.to(self.device)

                reps = self.model.base(x)                  # [B, D]
                reps = F.normalize(reps, p=2, dim=1)       # z: unit

                for i in range(len(y)):
                    if count >= num_samples:
                        break
                    Z_list.append(reps[i])
                    Y_list.append(int(y[i].item()))
                    count += 1

        if len(Z_list) == 0:
            return []

        Z = torch.stack(Z_list, dim=0)  # [N, D]
        Y = torch.tensor(Y_list, device=self.device, dtype=torch.long)  # [N]

        # ---------------------------
        # 2) build g per class
        # ---------------------------
        g_dict = {}

        if mode == "head":
            # head weight: [C, D]
            head_w = self.model.head.weight
            head_w = F.normalize(head_w, p=2, dim=1)
            # g per label
            unique = torch.unique(Y).tolist()
            for lbl in unique:
                if lbl < head_w.shape[0]:
                    g_dict[int(lbl)] = head_w[int(lbl)]

        elif mode == "mean":
            # local class mean of z
            unique = torch.unique(Y).tolist()
            for lbl in unique:
                mask = (Y == int(lbl))
                mu = Z[mask].mean(dim=0)
                mu = F.normalize(mu, p=2, dim=0)
                g_dict[int(lbl)] = mu

        elif mode == "global":
            # global_protos: dict lbl -> tensor [D] or [1,D] (cpu or gpu)
            if global_protos is None:
                return []
            for lbl, p in global_protos.items():
                t = p
                if not torch.is_tensor(t):
                    continue
                t = t.to(self.device)
                if t.dim() == 2:
                    t = t.squeeze(0)
                t = F.normalize(t, p=2, dim=0)
                g_dict[int(lbl)] = t

        else:
            raise ValueError(f"Unknown mode={mode}")

        # ---------------------------
        # 3) decompose each sample
        # ---------------------------
        with torch.no_grad():
            for i in range(Z.shape[0]):
                lbl = int(Y[i].item())
                if lbl not in g_dict:
                    continue
                z = Z[i]          # unit
                g = g_dict[lbl]   # unit

                cos_zg = torch.dot(z, g).clamp(-1.0, 1.0)  # cosine
                content = cos_zg * g                       # projection
                style = z - content                        # orthogonal residual

                extracted.append({
                    "client_id": self.id,
                    "label": lbl,
                    "content": content.detach().cpu().numpy(),
                    "style": style.detach().cpu().numpy(),
                    "cos_zg": float(cos_zg.item()),
                })

        return extracted

    
    def test_metrics(self, test_loader=None):
        if test_loader is None:
            test_loader = self.testloader
        self.model.eval()
        self.model.to(self.device)
        
        eval_protos = {}
        if hasattr(self, "personalized_protos") and self.personalized_protos:
            eval_protos = self.personalized_protos
        elif hasattr(self, "global_protos") and self.global_protos:
            eval_protos = self.global_protos
            
        P_matrix = None
        sorted_lbls = []
        
        if eval_protos:
            sorted_lbls = sorted(eval_protos.keys())
            P_matrix = torch.stack([eval_protos[l].to(self.device).squeeze() for l in sorted_lbls])
            P_matrix = F.normalize(P_matrix, dim=1)

        correct = 0
        total = 0
        y_prob_list = []
        y_true_list = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                if hasattr(self.model, 'base'): features_raw = self.model.base(x)
                else: features_raw = self.model(x)
                features_raw = features_raw.view(features_raw.size(0), -1)
                
                # 🎛️ [Conditional Logic] FiLM 적용 여부 확인 (Test Time)
                features_film = features_raw # 기본값
                
                if self.use_film and hasattr(self, "local_style") and self.local_style:
                    style_batch = []
                    for yy in y.tolist():
                        s = self.local_style.get(yy)
                        if isinstance(s, torch.Tensor): 
                            style_batch.append(s.squeeze(0).to(self.device))
                        else: 
                            style_batch.append(torch.zeros_like(features_raw[0]))
                    
                    if style_batch:
                        style_batch = torch.stack(style_batch, dim=0)
                        features_film = self.film(features_raw, style_batch)

                # Classifier Head
                if hasattr(self.model, 'head'): logits_cls = self.model.head(features_film)
                else: logits_cls = self.model.fc(features_film)
                
                probs = F.softmax(logits_cls, dim=1)
                y_prob_list.append(probs.cpu().numpy())
                
                y_onehot = F.one_hot(y, num_classes=self.num_classes)
                y_true_list.append(y_onehot.cpu().numpy())
                
                _, predicted = torch.max(probs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        if len(y_prob_list) == 0: return 0, 0, np.array([]), np.array([])
        return correct, total, np.concatenate(y_prob_list, axis=0), np.concatenate(y_true_list, axis=0)
