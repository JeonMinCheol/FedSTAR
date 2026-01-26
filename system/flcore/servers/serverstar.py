# fedstar_server.py
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.nn.functional as F

from flcore.servers.serverbase import Server
from flcore.clients.clientstar import clientstar
from utils.visual import * 

# =========================================================
# Improved Head-only Retrieval / Disentanglement Analysis
# =========================================================
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

def _safe_stack(db, key):
    """Return stacked np.array or None if key missing."""
    if not db or key not in db[0]:
        return None
    return np.array([d[key] for d in db], dtype=np.float32)


def _knn_stats(X, y_label, y_client, k=5, metric="cosine"):
    """
    X: (N, D)
    y_label: (N,)
    y_client: (N,)
    """
    N = X.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, N), metric=metric, algorithm="brute")
    nn.fit(X)
    dist, idx = nn.kneighbors(X, return_distance=True)

    # exclude self at position 0
    idx_k = idx[:, 1:1 + k]  # (N, k)

    same_label = (y_label[idx_k] == y_label[:, None]).mean()
    same_client = (y_client[idx_k] == y_client[:, None]).mean()

    # neighbor sets for overlap computation
    return same_label, same_client, idx_k


def _neighbor_overlap(idxA, idxB):
    """
    idxA, idxB: (N, k) neighbor indices
    Returns:
      jaccard_mean: mean Jaccard across samples
      top1_same: fraction where top-1 neighbor is same
      top5_overlap_mean: mean(|intersection|/k) across samples
    """
    N, k = idxA.shape
    jac = []
    top5 = []
    top1_same = (idxA[:, 0] == idxB[:, 0]).mean()

    for i in range(N):
        a = set(idxA[i].tolist())
        b = set(idxB[i].tolist())
        inter = len(a & b)
        union = len(a | b)
        jac.append(inter / union if union > 0 else 0.0)
        top5.append(inter / k)

    return float(np.mean(jac)), float(top1_same), float(np.mean(top5))


def _cos_stats(z, w):
    """
    z, w: (N, D)
    Returns (cos_raw_mean, cos_raw_absmean, cos_norm_mean, cos_norm_absmean)
    """
    # raw cosine (no pre-normalization)
    z_norm = np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
    w_norm = np.linalg.norm(w, axis=1, keepdims=True) + 1e-12
    cos_raw = (z * w).sum(axis=1) / (z_norm[:, 0] * w_norm[:, 0])

    # normalized cosine classifier view
    zhat = z / z_norm
    what = w / w_norm
    cos_norm = (zhat * what).sum(axis=1)

    return (
        float(np.mean(cos_raw)), float(np.mean(np.abs(cos_raw))),
        float(np.mean(cos_norm)), float(np.mean(np.abs(cos_norm)))
    )


def _linear_probe(X, y, name="probe", test_size=0.25, seed=0):
    """
    Simple linear probe with LogisticRegression.
    Returns: (acc, macro_f1) or (None, None) if probe fails.
    """
    try:
        # stratify when possible
        strat = y if (len(np.unique(y)) > 1) else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=strat
        )

        clf = LogisticRegression(
            max_iter=2000,
            solver="saga",        # works for multinomial + sparse-ish
            n_jobs=-1,
            multi_class="auto",
            C=1.0
        )
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        acc = accuracy_score(yte, pred)
        mf1 = f1_score(yte, pred, average="macro", zero_division=0)
        return float(acc), float(mf1)
    except Exception as e:
        print(f"[Probe:{name}] skipped due to error: {e}")
        return None, None

def _linear_probe_balanced(
    X, y,
    name="probe",
    min_per_class=2,      # 이 이상 샘플 있는 클래스만 사용
    max_classes=50,       # 너무 많으면 probe가 무거우니 제한
    per_class_cap=50,     # 클래스당 최대 샘플수 제한
    test_size=0.25,
    seed=0,
):
    """
    - y에서 count >= min_per_class 클래스만 유지
    - 클래스별로 per_class_cap까지 균등 샘플링
    - 남은 클래스가 2개 미만이면 probe 스킵
    """
    y = np.asarray(y)
    counts = Counter(y.tolist())
    valid = [c for c, n in counts.items() if n >= min_per_class]
    if len(valid) < 2:
        print(f"[Probe:{name}] skipped (valid classes < 2 after min_per_class={min_per_class}).")
        return None, None

    # 클래스 너무 많으면 상위(샘플 많은) 클래스만
    valid = sorted(valid, key=lambda c: counts[c], reverse=True)[:max_classes]

    # balanced indices
    idx = []
    rng = np.random.default_rng(seed)
    for c in valid:
        all_idx = np.where(y == c)[0]
        take = min(len(all_idx), per_class_cap)
        chosen = rng.choice(all_idx, size=take, replace=False)
        idx.append(chosen)
    idx = np.concatenate(idx)
    Xb = X[idx]
    yb = y[idx]

    # 이제 stratify 가능 (모든 클래스 count >= 2 보장됨)
    try:
        Xtr, Xte, ytr, yte = train_test_split(
            Xb, yb, test_size=test_size, random_state=seed, stratify=yb
        )
        clf = LogisticRegression(
            max_iter=2000, solver="saga", n_jobs=-1, multi_class="auto", C=1.0
        )
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        acc = accuracy_score(yte, pred)
        mf1 = f1_score(yte, pred, average="macro", zero_division=0)
        return float(acc), float(mf1)
    except Exception as e:
        print(f"[Probe:{name}] skipped due to error: {e}")
        return None, None


def run_disentanglement_analysis_head_only(
    clients,
    k_neighbors=5,
    metric="cosine",
    num_samples_per_client=50,
    probe_test_size=0.25,
    seed=0,
):
    print("\n" + "=" * 60)
    print(f"[Analysis] Head-only Retrieval / Disentanglement (k={k_neighbors})")
    print("=" * 60)

    # 1) Collect
    db = []
    target_total = 240          # 최소 200 넘기려고 여유 있게
    per_client = max(40, target_total // max(1, len(clients)))

    db = []
    for c in clients:
        data = c.extract_features_for_retrieval(num_samples=per_client)
        db.extend(data)

    if not db:
        print("[Analysis] No data extracted.")
        return

    # Required keys (existing in your current analysis)
    X_content = _safe_stack(db, "content")   # (N,D)
    X_style   = _safe_stack(db, "style")     # (N,D)
    y_label   = np.array([d["label"] for d in db], dtype=np.int64)
    y_client  = np.array([d["client_id"] for d in db], dtype=np.int64)

    N = len(db)
    print(f"[Analysis] N_used={N}, k={k_neighbors} | metric={metric}")

    # 2) KNN-based retrieval scores
    same_label_c, same_client_c, idx_c = _knn_stats(X_content, y_label, y_client, k=k_neighbors, metric=metric)
    same_label_s, same_client_s, idx_s = _knn_stats(X_style,   y_label, y_client, k=k_neighbors, metric=metric)

    jacc, top1_same, topk_overlap = _neighbor_overlap(idx_c, idx_s)

    style_norm = np.linalg.norm(X_style, axis=1)
    print(f"[Analysis] same-label@k   content={same_label_c:.4f} | style={same_label_s:.4f}")
    print(f"[Analysis] same-client@k  content={same_client_c:.4f} | style={same_client_s:.4f}")
    print(f"[Analysis] neighbor overlap Jaccard(content vs style) = {jacc:.4f}")
    print(f"[Analysis] overlap top1-same = {top1_same:.4f} | top{k_neighbors}-overlap(|∩|/k) = {topk_overlap:.4f}")
    print(f"[Analysis] ||style|| mean/median = {style_norm.mean():.6f} / {np.median(style_norm):.6f}")

    # 3) Geometry sanity (optional, only if keys exist)
    # Expect either:
    #   - "z" and "w_y" (recommended), or
    #   - "z" and "g" (fallback)
    z = _safe_stack(db, "z")
    w = _safe_stack(db, "w_y")
    if w is None:
        w = _safe_stack(db, "g")  # fallback name

    if (z is not None) and (w is not None):
        cos_raw_mean, cos_raw_abs, cos_norm_mean, cos_norm_abs = _cos_stats(z, w)
        print(f"[Analysis] cos_raw(z,w) mean/absmean = {cos_raw_mean:.4f} / {cos_raw_abs:.4f}")
        print(f"[Analysis] cos_norm(zhat,what) mean/absmean = {cos_norm_mean:.4f} / {cos_norm_abs:.4f}")
    else:
        print("[Analysis] cos(z,w) skipped (need db keys: 'z' and ('w_y' or 'g')).")

    # 4) Leakage probes (strongly recommended)
    acc_l, f1_l = _linear_probe_balanced(
        X_style, y_label,
        name="style->label",
        min_per_class=2,
        max_classes=50,
        per_class_cap=50,
        test_size=probe_test_size,
        seed=seed,
    )
    acc_c, f1_c = _linear_probe(X_style, y_client, name="style->client", test_size=probe_test_size, seed=seed)

    if acc_l is not None:
        print(f"[Probe] style->label  Acc={acc_l:.4f} | MacroF1={f1_l:.4f}  (LOW is better)")
    if acc_c is not None:
        print(f"[Probe] style->client Acc={acc_c:.4f} | MacroF1={f1_c:.4f}  (HIGH is better)")

    # 5) Return dict for logging
    out = {
        "N_used": int(N),
        "k": int(k_neighbors),
        "same_label@k_content": float(same_label_c),
        "same_label@k_style": float(same_label_s),
        "same_client@k_content": float(same_client_c),
        "same_client@k_style": float(same_client_s),
        "neighbor_overlap_jaccard(content,style)": float(jacc),
        "overlap_top1_same": float(top1_same),
        f"overlap_top{k_neighbors}_ratio": float(topk_overlap),
        "style_norm_mean": float(style_norm.mean()),
        "style_norm_median": float(np.median(style_norm)),
    }

    if (z is not None) and (w is not None):
        out.update({
            "cos_raw_mean": cos_raw_mean,
            "cos_raw_absmean": cos_raw_abs,
            "cos_norm_mean": cos_norm_mean,
            "cos_norm_absmean": cos_norm_abs,
        })

    if acc_l is not None:
        out.update({"probe_style_to_label_acc": acc_l, "probe_style_to_label_macroF1": f1_l})
    if acc_c is not None:
        out.update({"probe_style_to_client_acc": acc_c, "probe_style_to_client_macroF1": f1_c})

    return out

# =========================================================
# Server-side prototype aggregator (Transformer)
# =========================================================
class ServerPrototypeAggregator(nn.Module):
    def __init__(
        self,
        dataset,
        num_classes,
        embed_dim,
        num_layers=2,
        num_heads=4,
        dropout=0.05,
        num_clients=10,
    ):
        super().__init__()
        self.dataset = dataset
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = min(num_heads, embed_dim)

        self.client_embed = nn.Embedding(num_clients, embed_dim)
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=self.num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, CP, client_ids, all_labels, return_all=False):
        device = CP.device
        M, C, D = CP.shape

        id_embed = self.client_embed(client_ids).unsqueeze(1).expand(M, C, D)
        class_ids = torch.tensor(all_labels, device=device, dtype=torch.long)
        class_embed = self.class_embed(class_ids).unsqueeze(0).expand(M, C, D)

        x = self.norm(CP + id_embed + class_embed)
        x_trans = self.transformer(x)  # [M, C, D]

        if return_all:
            return x_trans

        global_proto = x_trans.mean(dim=0)  # [C, D]
        return {lbl: global_proto[j] for j, lbl in enumerate(all_labels)}


# =========================================================
# FedSTAR server
# =========================================================
class FedSTAR(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientstar)

        self.use_decompose_with_global = args.use_decompose_with_global
        self.use_transformer = args.use_transformer
        
        self.server_agg_steps = args.server_agg_steps
        self.server_agg_clip = args.server_agg_clip
        self.aggregator_learning_rate = args.aggregator_learning_rate

        self.global_protos = {}

        # Transformer 사용 시에만 모델 및 옵티마이저 초기화
        if self.use_transformer:
            embed_dim = self.clients[0].model.head.in_features
            self.aggregator = ServerPrototypeAggregator(
                self.dataset,
                self.num_classes,
                embed_dim,
                dropout=args.dropout,
                num_clients=self.num_clients,
                num_layers=2,
            ).to(self.device)

            self.agg_opt = torch.optim.SGD(
                self.aggregator.parameters(), lr=self.aggregator_learning_rate
            )
            print(f"[*] Aggregation Mode: Transformer Attention (Steps={self.server_agg_steps})")
        else:
            self.aggregator = None
            self.agg_opt = None
            print("[*] Aggregation Mode: Simple Average")

        # CPU oversubscription 방지
        torch.set_num_threads(1)
        self.max_parallel_clients = 6

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"[*] FedSTAR initialized. Parallel clients: {self.max_parallel_clients}")

    # -------------------- helpers --------------------
    @staticmethod
    def _ensure_2d_proto(t: torch.Tensor) -> torch.Tensor:
        """Make prototype shape [1, D]."""
        if t.dim() == 1:
            return t.unsqueeze(0)
        return t

    def _client_round(self, client):
        client.train()
        return client.id, client.collect_protos()
    
    def _update_client(self, client):
        client.set_protos(self.global_protos)
        if self.use_decompose_with_global:
            client.decompose_with_global(self.global_protos)

    # -------------------- main train --------------------
    def train(self):
        for round_num in range(self.global_rounds + 1):
            if round_num % self.eval_gap == 0:
                print(f"\n[Round {round_num}] Evaluate models")
                self.evaluate(round_num)

            start_time = time.time()
            selected_clients = self.select_clients()

            shared_list, client_ids, all_labels = [], [], []

            # 1. Collect prototypes
            with ThreadPoolExecutor(max_workers=self.max_parallel_clients) as executor:
                futures = [executor.submit(self._client_round, c) for c in selected_clients]
                for f in as_completed(futures):
                    try:
                        cid, cp = f.result()
                        if cp and "shared" in cp:
                            client_ids.append(cid)
                            shared_list.append(cp["shared"])  # dict[label] -> proto
                            all_labels.extend(list(cp["shared"].keys()))
                    except Exception as e:
                        print(f"[Warning] client thread failed: {e}")

            if len(shared_list) == 0:
                print(f"[Round {round_num}] No client prototypes collected. Skipping.")
                continue

            all_labels = sorted(set(all_labels))
            
            # 2. Aggregation Logic (Toggle)
            if self.use_transformer:
                # ==========================================
                # A) Transformer Aggregation
                # ==========================================
                
                # 2-1. Prepare Tensors
                first_dict = shared_list[0]
                first_key = next(iter(first_dict.keys()))
                D = self._ensure_2d_proto(first_dict[first_key]).shape[-1]

                CP = torch.stack([
                    torch.cat([
                        self._ensure_2d_proto(d.get(lbl, torch.zeros(D, device=self.device))).to(self.device)
                        for lbl in all_labels
                    ], dim=0)
                    for d in shared_list
                ], dim=0).to(self.device) # [M, C, D]

                CP = CP.detach()
                CP.requires_grad_(False)
                c_ids_tensor = torch.tensor(client_ids, device=self.device, dtype=torch.long)

                self.aggregator.train()
                last_x_trans = None
                last_loss = None

                # 2-2. Optimize Aggregator
                for s in range(self.server_agg_steps):
                    self.agg_opt.zero_grad(set_to_none=True)
                    x_trans = self.aggregator(CP, c_ids_tensor, all_labels, return_all=True)

                    present = (CP.abs().sum(dim=-1) > 1e-9).float() 
                    present_cnt = present.sum(dim=0).clamp(min=1.0)
                    global_shared = (x_trans * present.unsqueeze(-1)).sum(dim=0) / present_cnt.unsqueeze(-1)
                    
                    cos = F.cosine_similarity(global_shared.unsqueeze(0), CP, dim=-1)
                    loss = (1 - cos)[present.bool()].mean()
                    loss.backward()

                    if self.server_agg_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.aggregator.parameters(), self.server_agg_clip)
                    self.agg_opt.step()
                    last_x_trans = x_trans
                    last_loss = loss.detach()

                x_trans = last_x_trans

                # 2-3. Update Global Protos (Attention-based)
                with torch.no_grad():
                    M, C, D = x_trans.shape
                    present = (CP.abs().sum(dim=-1) > 1e-9)
                    label_tensor = torch.tensor(all_labels, device=self.device, dtype=torch.long)
                    class_vec = F.normalize(self.aggregator.class_embed(label_tensor), dim=-1)
                    
                    class_score = torch.einsum("mcd,cd->mc", x_trans, class_vec)
                    class_score = class_score.masked_fill(~present, -1e9)
                    attn_probs = F.softmax(class_score / (D ** 0.5), dim=0)
                    attn_probs = attn_probs * present.float()
                    attn_probs = attn_probs / (attn_probs.sum(dim=0, keepdim=True) + 1e-9)
                    
                    global_proto = (attn_probs.unsqueeze(-1) * x_trans).sum(dim=0)
                    new_protos = {lbl: global_proto[j].detach().cpu() for j, lbl in enumerate(all_labels)}
                    self.global_protos.update(new_protos)

                loss_val = float(last_loss.item()) if last_loss is not None else 0.0
                print(f"[Round {round_num}] Transformer Agg done. Loss: {loss_val:.6f}")

            else:
                # ==========================================
                # B) Simple Average Aggregation
                # ==========================================
                temp_storage = {} # label -> list of tensors

                for client_protos in shared_list:
                    for lbl, proto in client_protos.items():
                        # Device 이동 및 2D 보장
                        p = self._ensure_2d_proto(proto).to(self.device).detach()
                        if lbl not in temp_storage:
                            temp_storage[lbl] = []
                        temp_storage[lbl].append(p)
                
                new_protos = {}
                for lbl, p_list in temp_storage.items():
                    # [N, 1, D] -> [N, D] -> mean -> [D] -> [1, D]
                    # 단순 평균 (Simple Average)
                    mean_proto = torch.cat(p_list, dim=0).mean(dim=0, keepdim=True)
                    new_protos[lbl] = mean_proto.cpu()
                
                self.global_protos.update(new_protos)
                print(f"[Round {round_num}] Simple Aggregation done.")

            # 3. Broadcast
            for client in self.clients:
                self._update_client(client)
                
            dt = time.time() - start_time
            print(f"Time: {dt:.2f}s")

            # 서버가 가지고 있는 Global Prototypes (dict 형태)
            # 예: server.global_protos 
            # print("--- Final Disentanglement Analysis (HEAD-ONLY improved) ---")
            # stats = run_disentanglement_analysis_head_only(
            #     selected_clients,
            #     k_neighbors=5,
            #     metric="cosine",
            #     num_samples_per_client=3,
            #     probe_test_size=0.25,
            #     seed=0,
            # )
            # print("[Mode=head_only_improved]", stats)
