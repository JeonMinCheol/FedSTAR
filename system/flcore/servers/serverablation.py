import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from flcore.servers.serverbase import Server
from flcore.clients.clientablation import clientablation
import os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

class ServerPrototypeAggregator(nn.Module):
    def __init__(self, dataset, num_classes, embed_dim, num_layers=1, num_heads=8, dropout=0.05, num_clients=10):
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
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, CP, client_ids, all_labels, return_all=False):
        device = CP.device
        M, C, D = CP.shape
        id_embed = self.client_embed(client_ids).unsqueeze(1).expand(M, C, D)
        class_embed = self.class_embed(torch.tensor(all_labels, device=device)).unsqueeze(0).expand(M, C, D)
        x = self.norm(CP + id_embed + class_embed)
        x_trans = self.transformer(x)  # [M, C, D]
        if return_all:
            return x_trans
        global_proto = x_trans.mean(dim=0)  # [C, D]
        return {lbl: global_proto[j] for j, lbl in enumerate(all_labels)}
    
    def log_grad_stats(self):
        print("---- [Aggregator Gradients] ----")
        for name, p in self.named_parameters():
            if p.grad is not None:
                print(f"{name:50s} : {p.grad.abs().mean().item():.6e}")
        print("--------------------------------------")

# --- helper: pad to same C ---
def pad_to(tensor, target_C, dim=1):
    if tensor.shape[dim] < target_C:
        pad_shape = list(tensor.shape)
        pad_shape[dim] = target_C - tensor.shape[dim]
        pad = torch.zeros(*pad_shape, device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad], dim=dim)
    return tensor
        
class ablation(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientablation)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")

        embed_dim = self.clients[0].model.head.in_features
        self.aggregator = ServerPrototypeAggregator(
            self.dataset, self.num_classes, embed_dim,
            num_clients=self.num_clients, num_layers=1
        ).to(self.device)

        self.agg_opt = torch.optim.SGD(self.aggregator.parameters(), lr=self.learning_rate)
        self.global_protos = {}

        # CPU oversubscription 방지
        torch.set_num_threads(1)
        self.max_parallel_clients = 8
        print(f"[*] FedSTAR initialized. Parallel clients: {self.max_parallel_clients}")
    
    def _client_round(self, client):
        """클라이언트 한 명의 학습 + 프로토타입 수집"""
        client.train()
        return client.id, client.collect_protos()
    
    def train(self):
        for round_num in range(self.global_rounds + 1):
            if round_num % self.eval_gap == 0:
                print(f"\n[Round {round_num}] Evaluate models")
                self.evaluate(round_num)

            start_time = time.time()
            selected_clients = self.select_clients()
            shared_list, client_ids, all_labels = [], [], []

            # -------------------- Step 1. collect shared prototypes --------------------
            with ThreadPoolExecutor(max_workers=self.max_parallel_clients) as executor:
                futures = [executor.submit(self._client_round, c) for c in selected_clients]
                for f in as_completed(futures):
                    try:
                        cid, cp = f.result()
                        client_ids.append(cid)
                        shared_list.append(cp["shared"])
                        all_labels.extend(list(cp["shared"].keys()))
                    except Exception as e:
                        print(f"[Warning] client thread failed: {e}")

            all_labels = sorted(set(all_labels))
    
            D = list(shared_list[0].values())[0].shape[-1]
            CP = torch.stack([
                torch.cat([
                    (d.get(lbl, torch.zeros(1, D, device=self.device)).to(self.device))
                    for lbl in all_labels
                ], dim=0)
                for d in shared_list
            ]).to(self.device)

            client_ids = torch.tensor(client_ids, device=self.device, dtype=torch.long)

            # -------------------- Step 2. aggregate shared prototypes --------------------
            self.aggregator.train()
            self.agg_opt.zero_grad()

            x_trans = self.aggregator(CP, client_ids, all_labels, return_all=True)  # [M, C, D]
            global_shared = x_trans.mean(dim=0)
            total_loss = (1 - F.cosine_similarity(global_shared.unsqueeze(0), CP, dim=-1)).mean()
            total_loss.backward()
            self.agg_opt.step()

            # -------------------- Step 3. update global prototypes --------------------
            with torch.no_grad():
                # ① Fail-safe dimension alignment (x_trans와 class_embed만 고려)
                Cx = x_trans.shape[1]                                # aggregator output
                Ce = self.aggregator.class_embed.weight.shape[0]     # embedding table
                max_C = max(Cx, Ce)

                if Ce < max_C:
                    pad = torch.zeros(
                        max_C - Ce, self.aggregator.embed_dim,
                        device=self.aggregator.class_embed.weight.device,
                        dtype=self.aggregator.class_embed.weight.dtype
                    )
                    self.aggregator.class_embed.weight = nn.Parameter(
                        torch.cat([self.aggregator.class_embed.weight, pad], dim=0)
                    )

                x_trans = pad_to(x_trans, max_C, dim=1)              # [M, C, D]

                # (선택) 안정화용 정규화
                # x_trans = F.normalize(x_trans, dim=-1)
                self.aggregator.class_embed.weight.data = F.normalize(
                    self.aggregator.class_embed.weight.data, dim=-1
                )

                # ② Class-only attention score (Q·K dot)
                class_score = torch.einsum('mcd,cd->mc', x_trans, self.aggregator.class_embed.weight)  # [M, C]

                # ③ Attention weighting & aggregation (per-class normalize)
                attn_probs = F.softmax(class_score / (self.aggregator.embed_dim ** 0.5), dim=0)        # [M, C]
                attn_probs = attn_probs / (attn_probs.sum(dim=0, keepdim=True) + 1e-9)
                weights = attn_probs.unsqueeze(-1)                                                      # [M, C, 1]

                global_proto = (weights * x_trans).sum(dim=0)                                           # [C, D]

                # ④ Store global prototypes (truncate if more than all_labels)
                valid_C = len(all_labels)
                self.global_protos = {
                    lbl: global_proto[j].detach().cpu() for j, lbl in enumerate(all_labels[:valid_C])
                }
            
            # -------------------- Step 3. broadcast global prototypes --------------------
            for client in selected_clients:
                client.set_protos(self.global_protos)

            print(f"[Round {round_num}] Aggregation done. Time: {time.time() - start_time:.2f}s")