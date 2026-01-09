import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clientbase import Client

"""
클라이언트의 local style 벡터를 받아서,
원래 피쳐(x)를 per-client하게 modulation (γ, β 변환) 합니다.

γ: scaling, β: shifting.
"""
class StyleFiLM(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gamma = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh()   # 안정적 scaling
        )
        self.beta = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh()
        )

    def forward(self, x, style):
        """
        x: [B, D]
        style: [B, D] or [1, D]
        """
        gamma = self.gamma(style)
        beta = self.beta(style)
        return x * (1 + gamma) + beta

class ClientPrototypeGenerator(nn.Module):
    """
    - 클래스별 personal embedding 테이블(학습 파라미터)
    - 배치 임베딩에서 shared(mean) 산출
    - full = shared + personal
    """
    def __init__(self, model: nn.Module, num_classes: int):
        super().__init__()
        embed_dim = model.head.in_features
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.norm = nn.LayerNorm(embed_dim)

        # 클래스별 개인화 임베딩 파라미터
        self.personal_table = nn.Parameter(torch.zeros(num_classes, embed_dim))
        nn.init.zeros_(self.personal_table)

    def forward(self, emb_list: dict) -> dict:
        device = self.personal_table.device
        protos = {}

        for cls_id, embs_per_class in emb_list.items():
            if len(embs_per_class) == 0:
                continue

            embs = torch.stack(embs_per_class).to(device)
            embs = self.norm(embs)

            shared = embs.mean(dim=0, keepdim=True)
            personal = self.personal_table[cls_id].unsqueeze(0)
            full = shared + personal

            protos[cls_id] = {"shared": shared, "personal": personal, "full": full}

        return protos

class clientstar(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = args
        self.id = id
        self.device = torch.device(args.device)
        self.global_shared = {}
        self.global_protos = {}
        self.personalized_protos = {}
        self.local_personal_protos = {}

        # ===== NEW: prototype-regularization hyperparams =====
        self.lambda_proto_shared = self.args.lambda_proto_shared
        self.lambda_proto_pull = self.args.lambda_proto_pull
        self.num_classes = args.num_classes
        self.local_epochs = args.local_epochs

        self.criterion_ce = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()

        # self-attention 모듈
        self.proto_gen = ClientPrototypeGenerator(self.model, num_classes=self.num_classes).to(self.device)
        
        self.film = StyleFiLM(self.model.head.in_features).to(self.device)

        D = self.model.head.in_features
        self.alpha_gate = nn.Sequential(
            nn.Linear(4*D, D//4), nn.GELU(),
            nn.Linear(D//4, 1)
        ).to(self.device)

    def extract_features_for_retrieval(self, num_samples=20):
        """
        [분석 타겟 변경 버전]
        - Feature (z): model.base(x) [Raw Feature]
        - Target (g): model.head.weight [실제 분류기가 학습한 클래스 중심]
        """
        self.model.eval()
        self.model.to(self.device)
        
        trainloader = self.load_train_data(batch_size=num_samples)
        
        extracted_data = []
        count = 0
        eps = 1e-6

        with torch.no_grad():
            # 1. Head의 Weight를 정규화하여 준비 (이것이 진정한 Local Prototype)
            # shape: [Num_Classes, D]
            head_weights = F.normalize(self.model.head.weight, p=2, dim=1)

            for x, y in trainloader:
                if count >= num_samples: break

                x = x.to(self.device)
                y = y.to(self.device)
                
                # 2. Raw Feature 추출
                reps = self.model.base(x) # [B, D]
                
                # Feature도 정규화 (코사인 유사도 기반 분해를 위해 필수)
                reps_norm = F.normalize(reps, p=2, dim=1)

                for i in range(len(y)):
                    if count >= num_samples: break
                        
                    label = int(y[i].item())
                    
                    # z: 정규화된 Feature
                    z = reps_norm[i] 
                    
                    # g: 해당 클래스의 Head Weight (모델이 생각하는 정답 방향)
                    # 만약 Head가 Bias가 있다면 Bias는 무시하고 방향성만 봅니다.
                    if label < head_weights.shape[0]:
                        g = head_weights[label] 
                        
                        # -------------------------------------------------
                        # 3. Orthogonal Decomposition (변경 없음)
                        # -------------------------------------------------
                        # 이미 둘 다 정규화(norm=1) 되었으므로 dot_gg = 1
                        dot_zg = torch.dot(z, g) # 이것이 코사인 유사도
                        
                        # Content: g 방향 성분
                        content_vec = dot_zg * g
                        
                        # Style: g와 수직인 성분 (Residual)
                        style_vec = z - content_vec
                        
                        # 분석을 위해 Numpy 변환
                        # (분석 함수에서 다시 normalize 하더라도 여기서 값을 살려둠)
                        extracted_data.append({
                            "client_id": self.id,
                            "label": label,
                            "content": content_vec.cpu().numpy(),
                            "style": style_vec.cpu().numpy()
                        })
                        count += 1
        
        return extracted_data
    
    def _merge_alpha(self, g, p):
        g, p = g.view(1,-1), p.view(1,-1)
        z = torch.cat([g, p, (g-p).abs(), g*p], dim=-1)
        alpha = torch.sigmoid(self.alpha_gate(z))  # [1,1]
        return alpha

    def _get_proto_for_label(self, y_label: int):
        """
        personalized_protos > global_protos > None 순서로 프로토를 반환
        반환 텐서는 [D] or [1,D] 형태 모두 허용
        """
        dev = self.device
        if hasattr(self, "personalized_protos") and self.personalized_protos:
            p = self.personalized_protos.get(y_label)
            if isinstance(p, torch.Tensor):
                return p.to(dev).squeeze(0).detach()
        if isinstance(self.global_protos, dict) and len(self.global_protos) > 0:
            g = self.global_protos.get(y_label)
            if isinstance(g, torch.Tensor):
                return g.to(dev).squeeze(0).detach()
        return None

    def _build_proto_logits(self, reps: torch.Tensor, available_protos: dict):
        """
        reps: [B, D]
        available_protos: {cls_id: [1, D] or [D]}
        return: logits [B, C_avail] (C_avail = len(available_protos))
        방식: -||x - p_c||^2 * tau  (Prototypical classifier)
        """
        if not available_protos:
            return None, None
        dev = reps.device
        classes = sorted(available_protos.keys())
        protos = [available_protos[c].to(dev).squeeze(0).detach() for c in classes]  # [C, D]
        P = torch.stack(protos, dim=0)  # [C, D]
        # dist^2 = ||x||^2 + ||p||^2 - 2 x·p
        x2 = (reps**2).sum(dim=1, keepdim=True)        # [B,1]
        p2 = (P**2).sum(dim=1, keepdim=True).T         # [1,C]
        xp = reps @ P.T                                # [B,C]
        logits = 2 * xp - x2 - p2                       # [B,C]
        return logits, classes
    
    def train(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()
        start_time = time.time()

        for epoch in range(self.local_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                
                # 1. 원본 피쳐 추출
                reps_raw = self.model.base(x)  # [B, D] - FiLM 적용 전

                # -----------------------------------------------------------
                # 2. (NEW) 공유 분기 (Shared Branch) Loss 계산
                #    reps_raw를 global_protos 공간에 직접 정렬
                # -----------------------------------------------------------
                shared_loss = torch.tensor(0.0, device=self.device)
                # self.lambda_proto_shared > 0 이고, global_protos가 존재할 때만 실행
                if self.lambda_proto_shared > 0 and hasattr(self, "global_protos") and self.global_protos:
                    
                    # _build_proto_logits 함수를 reps_raw와 global_protos에 사용
                    plogits_s, classes_s = self._build_proto_logits(reps_raw, self.global_protos) 
                    
                    if plogits_s is not None:
                        # y를 해당 classes 인덱스로 매핑
                        class_to_idx_s = {c: i for i, c in enumerate(classes_s)}
                        y_mapped_s = []
                        for yy in y.tolist():
                            y_mapped_s.append(class_to_idx_s.get(yy, -1)) # get으로 안전하게
                        
                        y_mapped_s = torch.tensor(y_mapped_s, device=self.device, dtype=torch.long) #
                        mask_s = y_mapped_s >= 0
                        
                        if mask_s.any():
                            shared_loss = self.criterion_ce(plogits_s[mask_s], y_mapped_s[mask_s])

                # -----------------------------------------------------------
                # 3. (기존) 개인화 분기 (Personalized Branch)
                # -----------------------------------------------------------
                reps_personalized = reps_raw # 기본값
                if hasattr(self, "local_style") and self.local_style:
                    style_batch = []
                    for yy in y.tolist():
                        s = self.local_style.get(yy)
                        if isinstance(s, torch.Tensor):
                            style_batch.append(s.squeeze(0).to(self.device))
                        else:
                            style_batch.append(torch.zeros_like(reps_raw[0])) 
                    style_batch = torch.stack(style_batch, dim=0)
                    reps_personalized = self.film(reps_raw, style_batch) # personalized feature modulation
                
                # 3a. CE Loss (개인화된 로짓)
                logits = self.model.head(reps_personalized)
                ce_loss = self.criterion_ce(logits, y)

                # 3b. Prototype Pull Loss (개인화된 피쳐 -> 개인화된 프로토)
                pull_losses = []
                with torch.no_grad():
                    target_protos = []
                    valid_mask = []
                    for yy in y.tolist():
                        p = self._get_proto_for_label(yy)
                        if p is None:
                            valid_mask.append(False)
                            target_protos.append(torch.zeros_like(reps_personalized[0])) 
                        else:
                            valid_mask.append(True)
                            target_protos.append(p)
                    target_protos = torch.stack(target_protos, dim=0)
                    valid_mask = torch.tensor(valid_mask, device=self.device, dtype=torch.bool)

                if valid_mask.any():
                    rep_n = F.normalize(reps_personalized[valid_mask], dim=-1) 
                    proto_n = F.normalize(target_protos[valid_mask], dim=-1)
                    cos_pull = 1.0 - (rep_n * proto_n).sum(dim=-1)
                    pull_loss = cos_pull.mean()
                    pull_losses.append(pull_loss)

                total_loss = ce_loss
                
                if pull_losses:
                    total_loss = total_loss + self.lambda_proto_pull * torch.stack(pull_losses).mean()

                # (NEW) 공유 분기 로스 추가
                if shared_loss.item() > 0:
                    total_loss = total_loss + self.lambda_proto_shared * shared_loss

                # 5) 역전파
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                if self.learning_rate_decay and hasattr(self, 'learning_rate_scheduler'):
                    self.learning_rate_scheduler.step()

        # 시간·메트릭 저장
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_protos(self, global_shared):
        """
        Attention 기반 personalized prototype 생성:
        global_protos (shared)와 local_personal_protos를
        라벨 일치하는 부분만 attention-weighted merge.
        """
        self.global_protos = global_shared
        self.personalized_protos = {}

        if not hasattr(self, "local_personal_protos") or not self.local_personal_protos:
            return

        # ✅ 공통 라벨만 대상으로 merge
        shared_labels = set(self.global_protos.keys()) & set(self.local_personal_protos.keys())
        if not shared_labels:
            print(f"[Warning][Client {self.id}] No shared labels between local and global protos.")
            return

        for lbl in shared_labels:
            g = self.global_protos[lbl].to(self.device)
            p = self.local_personal_protos[lbl].to(self.device)

            alpha = self._merge_alpha(g,p)
            personalized = alpha * g + (1 - alpha) * p
            self.personalized_protos[lbl] = personalized.detach().cpu()

        # ✅ 나머지 (없는 클래스) 보정
        missing_labels = set(self.global_protos.keys()) - shared_labels
        for lbl in missing_labels:
            self.personalized_protos[lbl] = self.global_protos[lbl].detach().cpu()

    def collect_protos(self):
        self.model.eval()
        trainloader = self.load_train_data()
        emb_list = defaultdict(list)

        with torch.no_grad():
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                reps = self.model.base(x)  # [B, D]
                for i, label in enumerate(y):
                    emb_list[int(label)].append(reps[i])

        # ✅ 이제 emb_list를 전달
        local_proto_dict = self.proto_gen(emb_list)
        
        # 구조 분리
        shared = {k: v["shared"] for k, v in local_proto_dict.items()}
        personal = {k: v["personal"] for k, v in local_proto_dict.items()}

        # 저장해두기
        self.local_personal_protos = personal

        return {"shared": shared, "personal": personal}
    
    def decompose_with_global(self, global_shared):
        """
        global_shared: 서버가 보낸 클래스별 global prototype
        local_personal_protos: 클라이언트의 raw local prototype
        """
        self.local_style = {}
        self.local_content = {}
        eps = 1e-6

        for lbl, local_p in self.local_personal_protos.items():
            if lbl not in global_shared:
                continue
            g = global_shared[lbl].to(self.device)
            p = local_p.to(self.device)

            # 1️⃣ Projection: style factor = residual orthogonal to global
            proj = (p @ g.T) / (g @ g.T + eps) * g
            style = p - proj

            # 2️⃣ Normalization
            style = style / (style.norm(dim=-1, keepdim=True) + eps)

            self.local_content[lbl] = g.detach()
            self.local_style[lbl] = style.detach()

        return self.local_content, self.local_style
