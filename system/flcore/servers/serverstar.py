# fedstar_server.py
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.nn.functional as F

from flcore.servers.serverbase import Server
from flcore.clients.clientstar import clientstar

# =========================================================
# Server-side prototype aggregator (Transformer)
# =========================================================
class ServerPrototypeAggregator(nn.Module):
    """Aggregate per-class client prototypes with a transformer encoder."""

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
        # CP: [C, M, D]
        #   C = number of labels collected this round
        #   M = number of selected clients in this round
        #   D = prototype embedding dimension
        device = CP.device
        C, M, D = CP.shape

        id_embed = self.client_embed(client_ids).unsqueeze(0).expand(C, M, D)
        class_ids = torch.tensor(all_labels, device=device, dtype=torch.long)
        class_embed = self.class_embed(class_ids).unsqueeze(1).expand(C, M, D)

        x = CP + id_embed + class_embed
        x = self.norm(x)
        x_trans = self.transformer(x)  # [C, M, D]

        if return_all:
            return x_trans

        # Average across the client axis to get one prototype per class: [C, M, D] -> [C, D].
        global_proto = x_trans.mean(dim=1)
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
    def _ensure_2d_proto(self, t: torch.Tensor) -> torch.Tensor:
        """Return a prototype bank with shape [N, D]."""
        if t.dim() == 1:
            return t.unsqueeze(0)
        return t

    def _is_present(self, proto_tensor: torch.Tensor) -> torch.Tensor:
        """Mark which class-client slots contain a real prototype."""
        return proto_tensor.abs().sum(dim=-1) > 0.0

    def _client_round(self, client):
        client.train()
        return client.id, client.collect_protos()

    def _update_client(self, client):
        client.set_protos(self.global_protos)
        if self.use_decompose_with_global:
            client.decompose_with_global(self.global_protos)

    def _collect_shared_prototypes(self, selected_clients):
        shared_list, client_ids, all_labels = [], [], []

        with ThreadPoolExecutor(max_workers=self.max_parallel_clients) as executor:
            futures = [executor.submit(self._client_round, client) for client in selected_clients]
            for future in as_completed(futures):
                try:
                    client_id, client_payload = future.result()
                    if not client_payload or "shared" not in client_payload:
                        continue
                    client_ids.append(client_id)
                    shared_list.append(client_payload["shared"])
                    all_labels.extend(client_payload["shared"].keys())
                except Exception as exc:
                    print(f"[Warning] client thread failed: {exc}")

        return shared_list, client_ids, sorted(set(all_labels))

    def _build_class_client_tensor(self, shared_list, all_labels):
        first_dict = shared_list[0]
        first_key = next(iter(first_dict.keys()))
        embed_dim = self._ensure_2d_proto(first_dict[first_key]).shape[-1]

        zero_proto = torch.zeros(embed_dim, device=self.device)
        client_class = torch.stack(
            [
                torch.cat(
                    [
                        self._ensure_2d_proto(client_protos.get(label, zero_proto)).to(self.device)
                        for label in all_labels
                    ],
                    dim=0,
                )
                for client_protos in shared_list
            ],
            dim=0,
        ).to(self.device)

        # [M, C, D] -> [C, M, D]
        # We transpose so each class becomes one sequence and attention runs across clients.
        class_client = client_class.transpose(0, 1).detach()
        class_client.requires_grad_(False)
        return class_client

    def _aggregate_with_transformer(self, shared_list, client_ids, all_labels, round_num):
        CP = self._build_class_client_tensor(shared_list, all_labels)
        c_ids_tensor = torch.tensor(client_ids, device=self.device, dtype=torch.long)

        self.aggregator.train()
        last_x_trans = None
        last_loss = None

        for _ in range(self.server_agg_steps):
            self.agg_opt.zero_grad(set_to_none=True)
            present = self._is_present(CP)
            present_float = present.float()

            x_trans = self.aggregator(CP, c_ids_tensor, all_labels, return_all=True)

            # [C, M, D] * [C, M, 1] -> [C, D]
            # This keeps only valid class-client slots when forming the class prototype target.
            present_cnt = present_float.sum(dim=1).clamp(min=1.0)
            global_shared = (x_trans * present_float.unsqueeze(-1)).sum(dim=1) / present_cnt.unsqueeze(-1)

            # [C, D] -> [C, 1, D] so it can be compared against every client slot in CP: [C, M, D].
            cos = F.cosine_similarity(global_shared.unsqueeze(1), CP, dim=-1)
            loss = ((1 - cos) * present_float).sum() / present_float.sum().clamp(min=1.0)
            loss.backward()

            if self.server_agg_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.aggregator.parameters(), self.server_agg_clip)
            self.agg_opt.step()
            last_x_trans = x_trans
            last_loss = loss.detach()

        x_trans = last_x_trans

        with torch.no_grad():
            _, _, embed_dim = x_trans.shape
            present = self._is_present(CP)  # [C, M]
            label_tensor = torch.tensor(all_labels, device=self.device, dtype=torch.long)
            class_vec = F.normalize(self.aggregator.class_embed(label_tensor), dim=-1)  # [C, D]

            # [C, M, D] x [C, D] -> [C, M]
            # Each class scores the client slots that belong to the same class sequence.
            class_score = torch.einsum("cmd,cd->cm", x_trans, class_vec)
            class_score = class_score.masked_fill(~present, 0.0)
            attn_probs = F.softmax(class_score / (embed_dim ** 0.5), dim=1)
            attn_probs = attn_probs * present.float()
            attn_probs = attn_probs / (attn_probs.sum(dim=1, keepdim=True) + 1e-9)

            # [C, M, 1] * [C, M, D] -> [C, D]
            global_proto = (attn_probs.unsqueeze(-1) * x_trans).sum(dim=1)
            new_protos = {label: global_proto[idx].detach().cpu() for idx, label in enumerate(all_labels)}
            self.global_protos.update(new_protos)

        loss_val = float(last_loss.item()) if last_loss is not None else 0.0
        print(f"[Round {round_num}] Transformer Agg done. Loss: {loss_val:.6f}")

    def _aggregate_with_simple_average(self, shared_list, round_num):
        temp_storage = {}

        for client_protos in shared_list:
            for label, proto in client_protos.items():
                proto_2d = self._ensure_2d_proto(proto).to(self.device).detach()
                temp_storage.setdefault(label, []).append(proto_2d)

        new_protos = {}
        for label, proto_list in temp_storage.items():
            # [N, 1, D] -> [N, D] after cat, then mean over clients -> [1, D].
            mean_proto = torch.cat(proto_list, dim=0).mean(dim=0, keepdim=True)
            new_protos[label] = mean_proto.cpu()

        self.global_protos.update(new_protos)
        print(f"[Round {round_num}] Simple Aggregation done.")

    # -------------------- main train --------------------
    def train(self):
        for round_num in range(self.global_rounds + 1):
            if round_num % self.eval_gap == 0:
                print(f"\n[Round {round_num}] Evaluate models")
                self.evaluate(round_num)

            start_time = time.time()
            selected_clients = self.select_clients()
            shared_list, client_ids, all_labels = self._collect_shared_prototypes(selected_clients)

            if len(shared_list) == 0:
                print(f"[Round {round_num}] No client prototypes collected. Skipping.")
                continue

            if self.use_transformer:
                self._aggregate_with_transformer(shared_list, client_ids, all_labels, round_num)
            else:
                self._aggregate_with_simple_average(shared_list, round_num)

            # 3. Broadcast
            for client in self.clients:
                self._update_client(client)
                
            dt = time.time() - start_time
            print(f"Time: {dt:.2f}s")
