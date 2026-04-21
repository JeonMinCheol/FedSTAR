import time

import torch
import torch.nn.functional as F

from flcore.clients.clientstar import clientstar
from flcore.servers.serverbase import Server


class FedSTAR(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientstar)

        self.shared_dim = int(getattr(args, "shared_dim", 128))
        self.beta_ema = float(getattr(args, "anchor_beta_ema", 0.95))
        self.use_ema = bool(getattr(args, "use_ema", True))
        self.debug_losses = True
        self.anchor_eps = 1e-6
        self.min_var = 1e-4

        self.anchor_mu = torch.zeros(self.num_classes, self.shared_dim, device=self.device, dtype=torch.float32)
        self.anchor_var = torch.ones(self.num_classes, self.shared_dim, device=self.device, dtype=torch.float32)
        self.anchor_count = torch.zeros(self.num_classes, device=self.device, dtype=torch.float32)

        self.client_backbones = list(getattr(args, "client_backbones", []))

        torch.set_num_threads(1)

        print("[*] FedSTAR: direct shared statistics upload with count-based anchor aggregation")
        print(f"Join ratio / total clients: {self.join_ratio} / {self.num_clients}")
        if self.client_backbones:
            print(f"Client backbones: {self.client_backbones}")

    def _broadcast_anchors(self, clients):
        for client in clients:
            client.set_anchor_state(
                anchor_mu=self.anchor_mu,
                anchor_var=self.anchor_var,
                anchor_count=self.anchor_count,
            )

    def _client_round(self, client):
        client.train()
        return client.id, client.upload_payload, getattr(client, "latest_debug_stats", {})

    def _collect_client_packets(self, selected_clients):
        shared_list, client_ids, debug_stats = [], [], []
        for client in selected_clients:
            try:
                client_id, client_payload, client_debug = self._client_round(client)
            except Exception as exc:
                print(f"[Warning] client {client.id} failed: {exc}")
                continue
            if not client_payload or not client_payload.get("shared"):
                continue
            client_ids.append(client_id)
            shared_list.append(client_payload["shared"])
            if client_debug:
                debug_stats.append(client_debug)
        return shared_list, client_ids, debug_stats

    def _log_round_debug(self, round_num, debug_stats):
        if len(debug_stats) == 0:
            return

        def avg(key):
            return sum(float(item.get(key, 0.0)) for item in debug_stats) / float(len(debug_stats))

        print(
            f"[Round {round_num}] LocalDebug "
            f"cls={avg('avg_cls_loss'):.4f} | "
            f"shared_cls={avg('avg_shared_cls_loss'):.4f} | "
            f"align_w={avg('avg_align_weighted'):.4f} | "
            f"sep_w={avg('avg_sep_weighted'):.4f} | "
            f"style_w={avg('avg_private_style_weighted'):.4f} | "
            f"anchor_var={avg('anchor_var_mean'):.6f} | "
            f"class_sep={avg('class_sep'):.4f} | "
            f"pos_sim={avg('avg_pos_sim'):.4f} | "
            f"neg_sim={avg('avg_neg_sim'):.4f} | "
            f"anchor_ce={avg('avg_anchor_ce'):.4f} | "
            f"center_ce={avg('avg_center_anchor_ce'):.4f}"
        )

    @torch.no_grad()
    def _class_separation_from_means(self, mus: torch.Tensor) -> float:
        if mus.ndim == 1:
            mus = mus.unsqueeze(0)
        if mus.size(0) < 2:
            return 0.0

        mus = F.normalize(mus.to(dtype=torch.float32), p=2, dim=1, eps=self.anchor_eps)
        sim = mus @ mus.T
        eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
        nearest_cos = sim.masked_fill(eye, -1.0).max(dim=1).values
        return float((1.0 - nearest_cos.mean()).item())

    def _compute_weights(self, counts: torch.Tensor) -> torch.Tensor:
        return counts.clamp_min(self.anchor_eps)

    def _packet_to_shared_stats(self, packet):
        count = float(packet["count"])
        packet_mu = torch.as_tensor(packet["mu"], device=self.device, dtype=torch.float32)
        packet_var = torch.as_tensor(packet["var"], device=self.device, dtype=torch.float32)
        packet_mu = torch.nan_to_num(packet_mu, nan=0.0, posinf=0.0, neginf=0.0)
        packet_var = torch.nan_to_num(
            packet_var,
            nan=self.min_var,
            posinf=self.min_var,
            neginf=self.min_var,
        ).clamp_min(self.min_var)
        return count, packet_mu, packet_var

    def aggregate(self, shared_list, round_num):
        updated_classes = 0
        clients_per_class = []

        for c in range(self.num_classes):
            class_packets = []
            for client_stats in shared_list:
                if int(c) in client_stats:
                    class_packets.append(client_stats[int(c)])
            if len(class_packets) == 0:
                continue

            packet_stats = [self._packet_to_shared_stats(packet) for packet in class_packets]
            counts = torch.tensor(
                [packet_count for packet_count, _, _ in packet_stats],
                device=self.device,
                dtype=torch.float32,
            ).clamp_min(1.0)
            mus = torch.stack([packet_mu for _, packet_mu, _ in packet_stats], dim=0)
            vars_diag = torch.stack([packet_var for _, _, packet_var in packet_stats], dim=0)

            weights = self._compute_weights(counts)
            norm = weights.sum().clamp_min(self.anchor_eps)

            mu_hat = (weights.unsqueeze(1) * mus).sum(dim=0) / norm
            mu_hat = torch.nan_to_num(mu_hat, nan=0.0, posinf=0.0, neginf=0.0)

            centered = mus - mu_hat.unsqueeze(0)
            var_hat = (weights.unsqueeze(1) * (vars_diag + centered.pow(2))).sum(dim=0) / norm
            var_hat = torch.nan_to_num(
                var_hat,
                nan=self.min_var,
                posinf=self.min_var,
                neginf=self.min_var,
            ).clamp_min(self.min_var)

            ema_beta = self.beta_ema if self.use_ema else 0.0
            self.anchor_mu[c] = ema_beta * self.anchor_mu[c] + (1.0 - ema_beta) * mu_hat
            self.anchor_var[c] = ema_beta * self.anchor_var[c] + (1.0 - ema_beta) * var_hat
            self.anchor_var[c] = self.anchor_var[c].clamp_min(self.min_var)
            self.anchor_count[c] = counts.sum()

            updated_classes += 1
            clients_per_class.append(len(class_packets))

        mean_clients = (
            float(sum(clients_per_class)) / float(len(clients_per_class))
            if len(clients_per_class) > 0
            else 0.0
        )
        active_mask = self.anchor_count > 0
        anchor_sep = self._class_separation_from_means(self.anchor_mu[active_mask])
        print(
            f"[Round {round_num}] Anchor aggregation done. "
            f"updated_classes={updated_classes} | mean_clients_per_class={mean_clients:.4f} | "
            f"ema={self.use_ema} | "
            f"anchor_sep={anchor_sep:.4f}"
        )

    def train(self):
        self._broadcast_anchors(self.clients)

        for round_num in range(self.global_rounds + 1):
            if round_num % self.eval_gap == 0:
                print(f"\n[Round {round_num}] Evaluate models")
                self.evaluate(round_num)

            start_time = time.time()
            self.selected_clients = self.select_clients()
            self._broadcast_anchors(self.selected_clients)
            for client in self.selected_clients:
                client.set_round(round_num)

            shared_list, client_ids, debug_stats = self._collect_client_packets(self.selected_clients)

            if len(shared_list) == 0:
                print(f"[Round {round_num}] No client shared statistics collected. Skipping.")
                continue

            self._log_round_debug(round_num, debug_stats)
            self.aggregate(shared_list, round_num)
            self._broadcast_anchors(self.clients)
            for client in self.clients:
                client.decompose_with_global()

            dt = time.time() - start_time
            print(f"[Round {round_num}] active_clients={len(client_ids)} | time={dt:.2f}s")