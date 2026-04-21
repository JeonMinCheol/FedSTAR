import time

import numpy as np
import torch
import torch.nn.functional as F

from flcore.clients.clientbase import Client


class clientstar(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.device = torch.device(args.device)
        self.model.to(self.device)

        self.num_classes = int(args.num_classes)
        self.shared_dim = int(args.shared_dim)
        self.private_dim = int(getattr(args, "private_dim", args.shared_dim))
        self.use_private_branch = bool(args.use_private_branch)
        self.use_separation_loss = bool(args.use_separation_loss)
        self.normalize_shared_align = bool(getattr(args, "normalize_shared_align", True))
        self.use_anchor_softmax = bool(getattr(args, "use_anchor_softmax", True))

        self.lambda_align = float(args.lambda_align)
        self.lambda_sep = float(args.lambda_sep)
        self.lambda_private_style = float(getattr(args, "lambda_private_style", 0.0))
        self.shared_cls_weight = 1.0
        self.anchor_eps = 1e-6
        self.anchor_softmax_weight = float(getattr(args, "anchor_softmax_weight", 1.0))
        self.anchor_center_weight = float(getattr(args, "anchor_center_weight", 0.5))
        self.anchor_cosface_margin = float(getattr(args, "anchor_cosface_margin", 0.15))
        self.anchor_cosface_scale = float(getattr(args, "anchor_cosface_scale", 16.0))
        self.min_var = 1e-4
        self.warmup_rounds = 1

        self.use_amp = bool(getattr(args, "use_amp", self.device.type == "cuda"))
        self.amp_enabled = bool(self.use_amp and self.device.type == "cuda")
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        self.current_round = -1
        self.anchor_mu = torch.zeros(self.num_classes, self.shared_dim, device=self.device)
        self.anchor_var = torch.ones(self.num_classes, self.shared_dim, device=self.device)
        self.anchor_count = torch.zeros(self.num_classes, device=self.device)
        self.upload_payload = {"shared": {}}
        self.local_private_protos = {}
        self.local_content = {}
        self.local_style = {}
        self.latest_debug_stats = {}
        self.latest_class_sep = 0.0
        self.latest_align_debug = {
            "pos_sim": 0.0,
            "neg_sim": 0.0,
            "anchor_ce": 0.0,
            "center_anchor_ce": 0.0,
        }

    def set_round(self, round_num: int):
        self.current_round = int(round_num)

    def _move_input(self, x, y):
        if isinstance(x, list):
            x = x[0]
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True).long().view(-1)
        return x, y

    def _align_is_active(self) -> bool:
        return (
            self.lambda_align > 0
            and self.current_round >= self.warmup_rounds
            and bool((self.anchor_count > 0).any().item())
        )

    def _forward_batch(self, x):
        with torch.cuda.amp.autocast(enabled=self.amp_enabled):
            logits, _, z_s, z_p, shared_logits, private_logits = self.model(x, return_features=True)
        z_s = torch.nan_to_num(z_s.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        z_p = torch.nan_to_num(z_p.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        logits = torch.nan_to_num(
            logits.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        shared_logits = torch.nan_to_num(
            shared_logits.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        private_logits = torch.nan_to_num(
            private_logits.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        return logits, z_s, z_p, shared_logits, private_logits

    def _separation_loss(self, z_s: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
        if (not self.use_private_branch) or (not self.use_separation_loss) or z_p.numel() == 0:
            return z_s.new_zeros(())
        if z_s.size(0) <= 1:
            return z_s.new_zeros(())
        z_s_centered = z_s - z_s.mean(dim=0, keepdim=True)
        z_p_centered = z_p - z_p.mean(dim=0, keepdim=True)
        cross_cov = z_s_centered.T @ z_p_centered
        cross_cov = cross_cov / float(max(z_s.size(0) - 1, 1))
        return torch.mean(cross_cov.pow(2))

    def _shared_for_anchor(self, z_s: torch.Tensor) -> torch.Tensor:
        if not self.normalize_shared_align:
            return z_s
        return F.normalize(z_s, p=2, dim=1, eps=self.anchor_eps)

    def _alignment_loss(self, z_s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.latest_align_debug = {
            "pos_sim": 0.0,
            "neg_sim": 0.0,
            "anchor_ce": 0.0,
            "center_anchor_ce": 0.0,
        }
        if not self._align_is_active():
            return z_s.new_zeros(())

        valid = self.anchor_count.index_select(0, y) > 0
        if valid.sum().item() == 0:
            return z_s.new_zeros(())

        y_valid = y[valid]
        z_valid = self._shared_for_anchor(z_s[valid])
        target_mu = self.anchor_mu.index_select(0, y_valid).to(dtype=z_valid.dtype)
        target_mu = self._shared_for_anchor(target_mu)
        sq_dist = (z_valid - target_mu).pow(2).sum(dim=1)
        z_contrast = F.normalize(z_valid, p=2, dim=1, eps=self.anchor_eps)
        pos_anchor = F.normalize(target_mu, p=2, dim=1, eps=self.anchor_eps)
        pos_sim = torch.sum(z_contrast * pos_anchor, dim=1)

        align_vec = sq_dist
        neg_sim_mean = z_valid.new_zeros(())
        anchor_ce_mean = z_valid.new_zeros(())
        center_anchor_ce_mean = z_valid.new_zeros(())
        active_anchor_ids = torch.nonzero(self.anchor_count > 0, as_tuple=False).view(-1)

        if active_anchor_ids.numel() > 1:
            active_anchor_mu = self.anchor_mu.index_select(0, active_anchor_ids).to(dtype=z_valid.dtype)
            active_anchor_mu = F.normalize(active_anchor_mu, p=2, dim=1, eps=self.anchor_eps)
            sim_all = z_contrast @ active_anchor_mu.T
            same_class_mask = active_anchor_ids.unsqueeze(0) == y_valid.unsqueeze(1)

            neg_sim = sim_all.masked_fill(same_class_mask, -1.0).max(dim=1).values
            neg_sim_mean = neg_sim.mean()

            if self.use_anchor_softmax:
                target_positions = same_class_mask.to(dtype=torch.int64).argmax(dim=1)
                cosface_logits = sim_all.clone()
                cosface_logits = cosface_logits - (
                    self.anchor_cosface_margin * same_class_mask.to(dtype=cosface_logits.dtype)
                )
                cosface_logits = max(self.anchor_cosface_scale, self.anchor_eps) * cosface_logits
                anchor_ce = F.cross_entropy(cosface_logits, target_positions, reduction="none")
                align_vec = align_vec + self.anchor_softmax_weight * anchor_ce
                anchor_ce_mean = anchor_ce.mean()

            unique_classes, inverse = torch.unique(y_valid, sorted=True, return_inverse=True)
            center_sums = torch.zeros(
                unique_classes.size(0),
                z_valid.size(1),
                device=z_valid.device,
                dtype=z_valid.dtype,
            )
            center_counts = torch.zeros(
                unique_classes.size(0),
                device=z_valid.device,
                dtype=z_valid.dtype,
            )
            center_sums.index_add_(0, inverse, z_valid)
            center_counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=z_valid.dtype))
            class_centers = center_sums / center_counts.unsqueeze(1).clamp_min(1.0)
            class_centers = F.normalize(class_centers, p=2, dim=1, eps=self.anchor_eps)

            center_sim_all = class_centers @ active_anchor_mu.T
            center_same_class_mask = active_anchor_ids.unsqueeze(0) == unique_classes.unsqueeze(1)
            center_target_positions = center_same_class_mask.to(dtype=torch.int64).argmax(dim=1)
            center_logits = center_sim_all.clone()
            center_logits = center_logits - (
                self.anchor_cosface_margin * center_same_class_mask.to(dtype=center_logits.dtype)
            )
            center_logits = max(self.anchor_cosface_scale, self.anchor_eps) * center_logits
            center_anchor_ce_mean = F.cross_entropy(center_logits, center_target_positions, reduction="mean")

        self.latest_align_debug = {
            "pos_sim": float(pos_sim.mean().detach().item()),
            "neg_sim": float(neg_sim_mean.detach().item()),
            "anchor_ce": float(anchor_ce_mean.detach().item()),
            "center_anchor_ce": float(center_anchor_ce_mean.detach().item()),
        }

        return torch.mean(align_vec) + self.anchor_center_weight * center_anchor_ce_mean

    def _private_style_loss(self, z_p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if (not self.use_private_branch) or z_p.numel() == 0:
            return z_p.new_zeros(())
        if self.private_dim != self.shared_dim or len(self.local_style) == 0:
            return z_p.new_zeros(())

        style_targets = []
        valid_indices = []
        for idx, label in enumerate(y.tolist()):
            style = self.local_style.get(int(label))
            if style is None:
                continue
            if torch.linalg.norm(style).item() <= self.anchor_eps:
                continue
            valid_indices.append(idx)
            style_targets.append(style.to(device=z_p.device, dtype=z_p.dtype))

        if len(valid_indices) == 0:
            return z_p.new_zeros(())

        z_valid = F.normalize(z_p[valid_indices], p=2, dim=1, eps=self.anchor_eps)
        target_style = torch.stack(style_targets, dim=0)
        target_style = F.normalize(target_style, p=2, dim=1, eps=self.anchor_eps)
        return torch.mean(1.0 - torch.sum(z_valid * target_style, dim=1))

    def _loss_terms(
        self,
        logits: torch.Tensor,
        shared_logits: torch.Tensor,
        y: torch.Tensor,
        z_s: torch.Tensor,
        z_p: torch.Tensor,
    ):
        cls_loss = self.loss(logits, y)
        shared_cls_loss = self.loss(shared_logits, y)
        align_loss = self._alignment_loss(z_s, y)
        sep_loss = self._separation_loss(z_s, z_p)
        private_style_loss = self._private_style_loss(z_p, y)
        total_loss = (
            cls_loss
            + self.shared_cls_weight * shared_cls_loss
            + self.lambda_align * align_loss
            + self.lambda_sep * sep_loss
            + self.lambda_private_style * private_style_loss
        )
        return (
            total_loss,
            cls_loss,
            shared_cls_loss,
            align_loss,
            sep_loss,
            private_style_loss,
        )

    @torch.no_grad()
    def _anchor_var_mean(self) -> float:
        active_mask = self.anchor_count > 0
        if active_mask.any():
            return float(self.anchor_var[active_mask].mean().item())
        return float(self.anchor_var.mean().item())

    @torch.no_grad()
    def _class_separation_score(self, class_dist) -> float:
        if len(class_dist) < 2:
            return 0.0

        mus = torch.stack(
            [packet["mu"].to(device=self.device, dtype=torch.float32) for packet in class_dist.values()],
            dim=0,
        )
        mus = F.normalize(mus, p=2, dim=1, eps=self.anchor_eps)
        sim = mus @ mus.T
        eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
        nearest_cos = sim.masked_fill(eye, -1.0).max(dim=1).values
        return float((1.0 - nearest_cos.mean()).item())

    @torch.no_grad()
    def _estimate_class_distribution(self, data_loader=None):
        if data_loader is None:
            data_loader = self.load_train_data()

        was_training = self.model.training
        self.model.eval()

        sum_z = torch.zeros(self.num_classes, self.shared_dim, device=self.device, dtype=torch.float32)
        sum_z2 = torch.zeros(self.num_classes, self.shared_dim, device=self.device, dtype=torch.float32)
        count = torch.zeros(self.num_classes, device=self.device, dtype=torch.float32)
        track_private = self.use_private_branch and self.private_dim == self.shared_dim
        if track_private:
            sum_private = torch.zeros(
                self.num_classes,
                self.private_dim,
                device=self.device,
                dtype=torch.float32,
            )
        else:
            sum_private = None

        for x, y in data_loader:
            x, y = self._move_input(x, y)
            _, z_s, z_p, _, _ = self._forward_batch(x)
            z_s = self._shared_for_anchor(z_s)
            sum_z.index_add_(0, y, z_s)
            sum_z2.index_add_(0, y, z_s.pow(2))
            count.index_add_(0, y, torch.ones_like(y, dtype=count.dtype))
            if track_private and z_p.numel() > 0:
                z_p = F.normalize(z_p, p=2, dim=1, eps=self.anchor_eps)
                sum_private.index_add_(0, y, z_p)

        active_classes = torch.nonzero(count > 0, as_tuple=False).view(-1).tolist()
        class_dist = {}
        private_protos = {}
        for c in active_classes:
            mu = sum_z[c] / count[c].clamp_min(1.0)
            var = (sum_z2[c] / count[c].clamp_min(1.0)) - mu.pow(2)
            var = torch.nan_to_num(var, nan=self.min_var, posinf=self.min_var, neginf=self.min_var)
            var = var.clamp_min(self.min_var)
            class_dist[int(c)] = {
                "count": int(count[c].item()),
                "mu": mu.detach(),
                "var": var.detach(),
            }
            if track_private:
                proto = sum_private[c] / count[c].clamp_min(1.0)
                proto = F.normalize(proto.unsqueeze(0), p=2, dim=1, eps=self.anchor_eps).squeeze(0)
                private_protos[int(c)] = proto.detach()

        if was_training:
            self.model.train()
        return class_dist, private_protos

    @torch.no_grad()
    def collect_shared_stats(self, data_loader=None):
        class_dist, private_protos = self._estimate_class_distribution(data_loader)
        self.latest_class_sep = self._class_separation_score(class_dist)
        self.local_private_protos = {
            int(c): proto.to(dtype=torch.float32).detach().clone()
            for c, proto in private_protos.items()
        }
        shared_stats = {}

        for c, packet in class_dist.items():
            shared_stats[int(c)] = {
                "count": int(packet["count"]),
                "mu": packet["mu"].to(dtype=torch.float32).detach().cpu(),
                "var": packet["var"].to(dtype=torch.float32).clamp_min(self.min_var).detach().cpu(),
            }

        self.upload_payload = {"shared": shared_stats}
        return self.upload_payload

    @torch.no_grad()
    def decompose_with_global(self):
        self.local_content = {}
        self.local_style = {}
        if (not self.use_private_branch) or self.private_dim != self.shared_dim:
            return
        if len(self.local_private_protos) == 0:
            return

        for label, private_proto in self.local_private_protos.items():
            if int(label) >= self.num_classes:
                continue
            if float(self.anchor_count[int(label)].item()) <= 0:
                continue

            g = self.anchor_mu[int(label)].to(device=self.device, dtype=torch.float32)
            g = F.normalize(g.unsqueeze(0), p=2, dim=1, eps=self.anchor_eps).squeeze(0)
            p = private_proto.to(device=self.device, dtype=torch.float32)
            p = F.normalize(p.unsqueeze(0), p=2, dim=1, eps=self.anchor_eps).squeeze(0)

            proj = torch.dot(p, g) / (torch.dot(g, g) + self.anchor_eps) * g
            style = p - proj
            style_norm = torch.linalg.norm(style)
            if style_norm.item() <= self.anchor_eps:
                continue

            self.local_content[int(label)] = g.detach().clone()
            self.local_style[int(label)] = (style / (style_norm + self.anchor_eps)).detach().clone()

    def set_anchor_state(self, anchor_mu=None, anchor_var=None, anchor_count=None):
        if anchor_mu is None or anchor_var is None:
            self.anchor_mu.zero_()
            self.anchor_var.fill_(1.0)
            self.anchor_count.zero_()
            return

        self.anchor_mu = torch.as_tensor(anchor_mu, device=self.device, dtype=torch.float32).clone()
        self.anchor_var = torch.as_tensor(anchor_var, device=self.device, dtype=torch.float32).clone()
        if anchor_count is None:
            self.anchor_count = torch.ones(self.num_classes, device=self.device, dtype=torch.float32)
        else:
            self.anchor_count = torch.as_tensor(
                anchor_count,
                device=self.device,
                dtype=torch.float32,
            ).clone()

        self.anchor_mu = torch.nan_to_num(self.anchor_mu, nan=0.0, posinf=0.0, neginf=0.0)
        self.anchor_var = torch.nan_to_num(
            self.anchor_var,
            nan=self.min_var,
            posinf=self.min_var,
            neginf=self.min_var,
        ).clamp_min(self.min_var)
        self.anchor_count = torch.nan_to_num(self.anchor_count, nan=0.0, posinf=0.0, neginf=0.0)

    def train(self):
        start_time = time.time()
        self.model.to(self.device)
        trainloader = self.load_train_data()
        self.model.train()

        sum_cls = 0.0
        sum_shared_cls = 0.0
        sum_align_weighted = 0.0
        sum_sep_weighted = 0.0
        sum_private_style_weighted = 0.0
        sum_pos_sim = 0.0
        sum_neg_sim = 0.0
        sum_anchor_ce = 0.0
        sum_center_anchor_ce = 0.0
        batch_count = 0

        for _ in range(self.local_epochs):
            for x, y in trainloader:
                x, y = self._move_input(x, y)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                logits, z_s, z_p, shared_logits, _ = self._forward_batch(x)
                (
                    total_loss,
                    cls_loss,
                    shared_cls_loss,
                    align_loss,
                    sep_loss,
                    private_style_loss,
                ) = self._loss_terms(
                    logits,
                    shared_logits,
                    y,
                    z_s,
                    z_p,
                )

                batch_count += 1
                sum_cls += float(cls_loss.detach().item())
                sum_shared_cls += float(shared_cls_loss.detach().item())
                sum_align_weighted += float((self.lambda_align * align_loss).detach().item())
                sum_sep_weighted += float((self.lambda_sep * sep_loss).detach().item())
                sum_private_style_weighted += float(
                    (self.lambda_private_style * private_style_loss).detach().item()
                )
                sum_pos_sim += float(self.latest_align_debug.get("pos_sim", 0.0))
                sum_neg_sim += float(self.latest_align_debug.get("neg_sim", 0.0))
                sum_anchor_ce += float(self.latest_align_debug.get("anchor_ce", 0.0))
                sum_center_anchor_ce += float(self.latest_align_debug.get("center_anchor_ce", 0.0))

                self.optimizer.zero_grad(set_to_none=True)
                if not torch.isfinite(total_loss):
                    continue

                if self.amp_enabled:
                    self.grad_scaler.scale(total_loss).backward()
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    total_loss.backward()
                    self.optimizer.step()

        self.collect_shared_stats(trainloader)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.latest_debug_stats = {
            "avg_cls_loss": sum_cls / max(batch_count, 1),
            "avg_shared_cls_loss": sum_shared_cls / max(batch_count, 1),
            "avg_align_weighted": sum_align_weighted / max(batch_count, 1),
            "avg_sep_weighted": sum_sep_weighted / max(batch_count, 1),
            "avg_private_style_weighted": sum_private_style_weighted / max(batch_count, 1),
            "anchor_var_mean": self._anchor_var_mean(),
            "class_sep": float(self.latest_class_sep),
            "avg_pos_sim": sum_pos_sim / max(batch_count, 1),
            "avg_neg_sim": sum_neg_sim / max(batch_count, 1),
            "avg_anchor_ce": sum_anchor_ce / max(batch_count, 1),
            "avg_center_anchor_ce": sum_center_anchor_ce / max(batch_count, 1),
        }

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def test_metrics(self, test_loader=None):
        if test_loader is None:
            test_loader = self.load_test_data()

        self.model.eval()
        self.model.to(self.device)

        correct = 0
        total = 0
        y_prob_list = []
        y_true_list = []

        with torch.no_grad():
            for x, y in test_loader:
                x, y = self._move_input(x, y)
                logits, _, _, _, _ = self._forward_batch(x)
                probs = F.softmax(logits, dim=1)

                y_prob_list.append(probs.detach().cpu().numpy())
                y_true_list.append(F.one_hot(y, num_classes=self.num_classes).detach().cpu().numpy())
                correct += (torch.argmax(probs, dim=1) == y).sum().item()
                total += y.size(0)

        if len(y_prob_list) == 0:
            return 0, 0, np.array([]), np.array([])
        return correct, total, np.concatenate(y_prob_list, axis=0), np.concatenate(y_true_list, axis=0)

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()
        self.model.to(self.device)

        train_correct = 0
        train_num = 0
        losses = 0.0
        with torch.no_grad():
            for x, y in trainloader:
                x, y = self._move_input(x, y)
                logits, _, _, _, _ = self._forward_batch(x)
                cls_loss = self.loss(logits, y)

                train_num += y.shape[0]
                losses += cls_loss.item() * y.shape[0]
                train_correct += (torch.argmax(logits, dim=1) == y).sum().item()

        return losses, train_num, train_correct
