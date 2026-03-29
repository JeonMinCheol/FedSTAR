import time

import torch

from flcore.servers.serverbase import Server
from flcore.clients.clientstar import clientstar


class FedSTAR(Server):
    """
    Core server for prototype personalization.

    Keeps only:
      - model aggregation
      - simple prototype aggregation
      - prototype broadcast
    """

    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientstar)

        self.global_protos = {}

        torch.set_num_threads(1)

        print("[*] Core FedSTAR: simple prototype averaging")
        print(f"Join ratio / total clients: {self.join_ratio} / {self.num_clients}")

    def _client_round(self, client):
        client.train()
        return client.id, client.protos

    def _collect_client_packets(self, selected_clients):
        shared_list, client_ids = [], []
        for client in selected_clients:
            try:
                client_id, client_payload = self._client_round(client)
            except Exception as exc:
                print(f"[Warning] client {client.id} failed: {exc}")
                continue
            if not client_payload or not client_payload.get("shared"):
                continue
            client_ids.append(client_id)
            shared_list.append(client_payload["shared"])
        return shared_list, client_ids

    def aggregate(self, shared_list, round_num):
        proto_sums = {}
        proto_counts = {}

        for client_protos in shared_list:
            for label, proto in client_protos.items():
                proto_vec = proto.to(self.device).detach().view(-1).to(dtype=torch.float32)
                proto_vec = torch.nan_to_num(proto_vec, nan=0.0, posinf=0.0, neginf=0.0)
                if not torch.isfinite(proto_vec).all():
                    continue
                label = int(label)
                if label not in proto_sums:
                    proto_sums[label] = proto_vec.clone()
                    proto_counts[label] = 1
                else:
                    proto_sums[label].add_(proto_vec)
                    proto_counts[label] += 1

        new_global = dict(self.global_protos)
        for label, proto_sum in proto_sums.items():
            proto = proto_sum / max(proto_counts[label], 1)
            proto = torch.nan_to_num(proto, nan=0.0, posinf=0.0, neginf=0.0)
            if torch.isfinite(proto).all():
                new_global[label] = proto.detach()

        self.global_protos = new_global

        mean_clients = (
            float(sum(proto_counts.values())) / float(len(proto_counts))
            if proto_counts else 0.0
        )
        print(
            f"[Round {round_num}] Simple proto aggregation done. "
            f"classes={len(self.global_protos)} | mean_clients_per_class={mean_clients:.4f}"
        )

    def train(self):
        for round_num in range(self.global_rounds + 1):
            if round_num % self.eval_gap == 0:
                print(f"\n[Round {round_num}] Evaluate models")
                self.evaluate(round_num)

            start_time = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            for client in self.clients:
                client.set_protos(self.global_protos)
            for client in self.selected_clients:
                client.set_round(round_num)

            shared_list, client_ids = self._collect_client_packets(self.selected_clients)

            if len(shared_list) == 0:
                print(f"[Round {round_num}] No client prototypes collected. Skipping.")
                continue

            self.receive_models()
            if len(self.uploaded_models) > 0:
                self.aggregate_parameters()
            self.aggregate(shared_list, round_num)

            for client in self.clients:
                client.set_parameters(self.global_model)
                client.set_protos(self.global_protos)

            dt = time.time() - start_time
            print(f"[Round {round_num}] active_clients={len(client_ids)} | time={dt:.2f}s")
