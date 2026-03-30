import math
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from flcore.clients.clientbase import Client


class clientstar(Client):
    """
    Core prototype-personalization client.

    Current paper-facing view:
      1) build local class prototypes m_{k,c}
      2) estimate a client-side interpretation map A_k : p_c -> \tilde p_{k,c}
      3) build personalized anchors q_{k,c} by convexly mixing interpreted global
         and local prototypes
      4) train with h_g(z) + lambda * r(z, q_k) and support-restricted proto CE
      5) upload raw local prototypes only
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.device = torch.device(args.device)
        self.model.to(self.device)

        self.num_classes = int(args.num_classes)
        self.local_epochs = int(args.local_epochs)
        self.embed_dim = int(self.model.head.in_features)

        # Core public knobs.
        legacy_proto_gain = float(getattr(args, "proto_gain", 1.0))
        self.proto_resid_w = float(getattr(args, "proto_resid_w", 1.0))
        self.proto_start_round = int(
            getattr(args, "proto_train_start_round", getattr(args, "proto_start_round", 5))
        )
        self.proto_warmup_epochs = int(
            getattr(args, "proto_personal_warmup_epochs", getattr(args, "proto_warmup_epochs", 1))
        )
        self.proto_temp = float(getattr(args, "proto_temp", 0.10))
        self.proto_cls_w = float(getattr(args, "proto_cls_w", legacy_proto_gain))
        self.proto_logit_w = float(getattr(args, "proto_logit_w", 2.0 * legacy_proto_gain))

        # Fixed internals.
        self.proto_count_tau = float(getattr(args, "proto_count_tau", 10.0))
        self.proto_error_tau = float(getattr(args, "proto_error_tau", 20.0))
        self.proto_logit_scale_cap = float(getattr(args, "proto_logit_scale_cap", 3.0))
        self.proto_eps = float(getattr(args, "proto_eps", 1e-6))
        self.proto_state_clip = float(getattr(args, "proto_state_clip", 50.0))
        self.proto_upload_norm = float(getattr(args, "proto_upload_norm", 50.0))
        self.proto_grad_clip = float(getattr(args, "proto_grad_clip", 5.0))
        self.use_core_personalization = bool(getattr(args, "use_core_personalization", True))
        self.use_affine_align = bool(getattr(args, "use_affine_align", getattr(args, "use_film", True)))
        self.use_residual_anchor = bool(getattr(args, "use_residual_anchor", True))
        self.use_proto_ce_loss = bool(getattr(args, "use_proto_ce_loss", True))

        self.use_amp = bool(getattr(args, "use_amp", self.device.type == "cuda"))
        self.amp_enabled = bool(self.use_amp and self.device.type == "cuda")
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        self.current_round = -1
        self.global_protos: Dict[int, torch.Tensor] = {}
        self.local_protos_cache: Dict[int, torch.Tensor] = {}
        self.local_class_counts: Dict[int, int] = {}
        self.protos = {"shared": {}}
        self.align_R = torch.eye(self.embed_dim, device=self.device, dtype=torch.float32)
        self.align_b = torch.zeros(self.embed_dim, device=self.device, dtype=torch.float32)
        self.align_overlap = 0
        self.align_fit_error = 0.0
        self.local_radius = 0.0
        self.personalization_stats = {}

    def set_round(self, round_num: int):
        self.current_round = int(round_num)

    @torch.no_grad()
    def _sanitize_vec(self, x, clip: Optional[float] = None) -> torch.Tensor:
        if torch.is_tensor(x):
            v = x.detach().to(self.device, dtype=torch.float32).view(-1)
        else:
            v = torch.as_tensor(x, device=self.device, dtype=torch.float32).view(-1)
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if clip is not None:
            v = v.clamp(min=-clip, max=clip)
        return v

    @torch.no_grad()
    def _clip_vec_norm(self, v: torch.Tensor, max_norm: float) -> torch.Tensor:
        norm = torch.norm(v, p=2)
        if torch.isfinite(norm) and norm.item() > float(max_norm):
            v = v * (float(max_norm) / (norm + self.proto_eps))
        return v

    def _should_use_personalized(self) -> bool:
        return bool(self.global_protos) and (
            self.current_round < 0 or self.current_round >= self.proto_start_round
        )

    @torch.no_grad()
    def _build_local_class_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.num_classes, device=self.device, dtype=torch.bool)
        for c, n in self.local_class_counts.items():
            if 0 <= int(c) < self.num_classes and int(n) > 0:
                mask[int(c)] = True
        return mask

    @torch.no_grad()
    def _reset_affine_aligner(self):
        self.align_R = torch.eye(self.embed_dim, device=self.device, dtype=torch.float32)
        self.align_b = torch.zeros(self.embed_dim, device=self.device, dtype=torch.float32)
        self.align_overlap = 0
        self.align_fit_error = 0.0

    def _apply_affine_align(self, features: torch.Tensor, enabled: bool = True) -> torch.Tensor:
        if (not enabled) or (not self.use_affine_align) or (not self.use_core_personalization):
            return features

        # A_k(v) = v R_k + b_k
        R = self.align_R.to(device=features.device, dtype=features.dtype)
        b = self.align_b.to(device=features.device, dtype=features.dtype)
        if features.dim() == 1:
            aligned = torch.matmul(features, R) + b
        else:
            aligned = torch.matmul(features, R) + b.unsqueeze(0)
        return torch.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def _estimate_affine_aligner(self, local_protos: Optional[Dict[int, torch.Tensor]] = None):
        if local_protos is None:
            local_protos = self.local_protos_cache

        if (
            (not self.use_core_personalization)
            or (not self.use_affine_align)
            or (not self.global_protos)
            or (not local_protos)
        ):
            self._reset_affine_aligner()
            return

        overlap = sorted(set(local_protos.keys()) & set(self.global_protos.keys()))
        if len(overlap) == 0:
            self._reset_affine_aligner()
            return

        # X[c] = p_c, Y[c] = m_{k,c} for classes shared by local/global banks.
        X = torch.stack(
            [self._sanitize_vec(self.global_protos[c], clip=self.proto_state_clip) for c in overlap],
            dim=0,
        )
        Y = torch.stack(
            [self._sanitize_vec(local_protos[c], clip=self.proto_state_clip) for c in overlap],
            dim=0,
        )
        w = torch.tensor(
            [float(self.local_class_counts.get(c, 1)) for c in overlap],
            device=self.device,
            dtype=torch.float32,
        ).clamp_min(1.0)
        w = w / w.sum().clamp_min(self.proto_eps)

        # x_mean = sum_c w_c p_c, y_mean = sum_c w_c m_{k,c}
        x_mean = (w.unsqueeze(1) * X).sum(dim=0)
        y_mean = (w.unsqueeze(1) * Y).sum(dim=0)
        Xc = X - x_mean.unsqueeze(0)
        Yc = Y - y_mean.unsqueeze(0)

        if len(overlap) == 1:
            R = torch.eye(self.embed_dim, device=self.device, dtype=torch.float32)
        else:
            # Weighted orthogonal Procrustes:
            # M = X_c^T W Y_c, M = U S V^T, R_k = U V^T
            Xw = Xc * torch.sqrt(w).unsqueeze(1)
            Yw = Yc * torch.sqrt(w).unsqueeze(1)
            M = torch.matmul(Xw.t(), Yw)
            U, _, Vh = torch.linalg.svd(M, full_matrices=False)
            R = torch.matmul(U, Vh)
            if torch.linalg.det(R).item() < 0:
                Vh[-1, :] *= -1.0
                R = torch.matmul(U, Vh)

        # b_k = y_mean - x_mean R_k
        b = y_mean - torch.matmul(x_mean, R)
        # fit_error = mean_c || A_k(p_c) - m_{k,c} ||_2
        aligned_X = torch.matmul(X, R) + b.unsqueeze(0)
        fit_error = torch.norm(aligned_X - Y, p=2, dim=1).mean()

        self.align_R = torch.nan_to_num(R.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        self.align_b = torch.nan_to_num(b.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        self.align_overlap = len(overlap)
        self.align_fit_error = float(fit_error.item()) if torch.isfinite(fit_error) else 0.0

    @torch.no_grad()
    def _build_interpreted_global_prototypes(self) -> Dict[int, torch.Tensor]:
        interpreted = {}
        use_interp = (
            self.use_core_personalization
            and self.use_affine_align
            and self._should_use_personalized()
            and self.align_overlap > 0
        )
        for c, proto in self.global_protos.items():
            vec = self._sanitize_vec(proto, clip=self.proto_state_clip)
            if use_interp:
                vec = self._apply_affine_align(vec, enabled=True)
            vec = self._clip_vec_norm(vec, self.proto_upload_norm)
            if torch.isfinite(vec).all():
                interpreted[int(c)] = vec.detach()
        return interpreted

    @torch.no_grad()
    def _build_local_prototypes(self, data_loader=None) -> Tuple[Dict[int, torch.Tensor], Dict[int, int]]:
        if data_loader is None:
            data_loader = self.load_train_data()

        was_training = self.model.training
        self.model.eval()

        feat_sums = torch.zeros(self.num_classes, self.embed_dim, device=self.device, dtype=torch.float32)
        feat_counts = torch.zeros(self.num_classes, device=self.device, dtype=torch.float32)

        for x, y in data_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True).long().view(-1)
            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                rep = self.model.base(x)
            rep = rep.to(dtype=torch.float32)
            if rep.ndim == 1:
                rep = rep.unsqueeze(0)
            rep = torch.nan_to_num(rep, nan=0.0, posinf=0.0, neginf=0.0)
            rep = rep.clamp(min=-self.proto_state_clip, max=self.proto_state_clip)
            valid = torch.isfinite(rep).all(dim=1)
            if valid.sum().item() == 0:
                continue
            rep = rep[valid]
            y = y[valid]
            feat_sums.index_add_(0, y, rep)
            feat_counts.index_add_(0, y, torch.ones_like(y, dtype=feat_counts.dtype))

        local_protos: Dict[int, torch.Tensor] = {}
        counts: Dict[int, int] = {}
        active_ids = torch.nonzero(feat_counts > 0, as_tuple=False).view(-1).tolist()
        for c in active_ids:
            # m_{k,c} = (1 / n_{k,c}) * sum_{i:y_i=c} f_k(x_i)
            proto = feat_sums[c] / feat_counts[c].clamp_min(1.0)
            proto = torch.nan_to_num(proto, nan=0.0, posinf=0.0, neginf=0.0)
            proto = self._clip_vec_norm(proto, self.proto_upload_norm)
            if torch.isfinite(proto).all():
                local_protos[int(c)] = proto.detach()
                counts[int(c)] = int(feat_counts[c].item())

        self.local_protos_cache = local_protos
        self.local_class_counts = counts
        if was_training:
            self.model.train()
        return local_protos, counts

    @torch.no_grad()
    def _compute_local_radius(self, data_loader=None) -> float:
        if data_loader is None:
            data_loader = self.load_train_data()
        if not self.local_protos_cache:
            return 0.0

        was_training = self.model.training
        self.model.eval()

        proto_bank = torch.zeros(
            self.num_classes, self.embed_dim, device=self.device, dtype=torch.float32
        )
        support_mask = torch.zeros(self.num_classes, device=self.device, dtype=torch.bool)
        for c, proto in self.local_protos_cache.items():
            if 0 <= int(c) < self.num_classes:
                proto_bank[int(c)] = F.normalize(
                    self._sanitize_vec(proto, clip=self.proto_state_clip),
                    p=2,
                    dim=0,
                    eps=self.proto_eps,
                )
                support_mask[int(c)] = True

        rho_k = 0.0
        for x, y in data_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True).long().view(-1)
            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                rep = self.model.base(x)
            rep = F.normalize(
                torch.nan_to_num(rep.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0),
                p=2,
                dim=1,
                eps=self.proto_eps,
            )
            valid = support_mask[y]
            if valid.sum().item() == 0:
                continue
            rep = rep[valid]
            y = y[valid]
            target_proto = proto_bank.index_select(0, y)
            dist = torch.norm(rep - target_proto, p=2, dim=1)
            if dist.numel() > 0:
                rho_k = max(rho_k, float(dist.max().item()))

        if was_training:
            self.model.train()
        return rho_k

    @torch.no_grad()
    def _build_anchor_bank(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.global_protos:
            return None, None
        if self.current_round >= 0 and self.current_round < self.proto_start_round:
            return None, None
        if self.use_core_personalization and not self.local_protos_cache:
            return None, None

        interpreted_global = self._build_interpreted_global_prototypes()
        anchors = []
        weights = []
        direct_errs = []
        interp_errs = []
        anchor_errs = []
        stat_weights = []

        for c in range(self.num_classes):
            gp = interpreted_global.get(c)
            lp = self.local_protos_cache.get(c)
            n_c = float(self.local_class_counts.get(c, 0))
            if gp is None:
                anchors.append(torch.zeros(self.embed_dim, device=self.device))
                weights.append(0.0)
                continue

            g_vec = self._sanitize_vec(gp, clip=self.proto_state_clip)
            count_conf = n_c / (n_c + self.proto_count_tau) if n_c > 0 else 0.0
            l_vec = self._sanitize_vec(lp, clip=self.proto_state_clip) if lp is not None else None
            if self.use_core_personalization and self.use_residual_anchor and lp is not None and n_c > 0:
                beta = max(0.0, min(1.0, self.proto_resid_w * count_conf))
                # q_{k,c} = (1 - beta_{k,c}) \tilde p_{k,c} + beta_{k,c} m_{k,c}
                anchor = (1.0 - beta) * g_vec + beta * l_vec
                anchor = self._clip_vec_norm(anchor, self.proto_upload_norm)
                weight = count_conf
            else:
                # Baseline ablation: q_{k,c} = \tilde p_{k,c}
                anchor = self._clip_vec_norm(g_vec, self.proto_upload_norm)
                weight = count_conf

            if l_vec is not None and n_c > 0:
                direct_g = self._sanitize_vec(self.global_protos[c], clip=self.proto_state_clip)
                direct_errs.append(float(torch.norm(direct_g - l_vec, p=2).item()))
                interp_errs.append(float(torch.norm(g_vec - l_vec, p=2).item()))
                anchor_errs.append(float(torch.norm(anchor - l_vec, p=2).item()))
                stat_weights.append(max(n_c, 1.0))

            anchors.append(anchor)
            weights.append(weight)

        anchor_bank = torch.stack(anchors, dim=0)
        proto_weights = torch.tensor(weights, device=self.device, dtype=anchor_bank.dtype)
        if len(stat_weights) > 0:
            w = np.asarray(stat_weights, dtype=np.float64)
            w = w / max(np.sum(w), 1.0)
            support_classes = sorted(
                set(self.local_protos_cache.keys()) & set(self.global_protos.keys())
            )
            if len(support_classes) >= 2:
                proto_stack = torch.stack(
                    [
                        F.normalize(
                            self._sanitize_vec(self.local_protos_cache[c], clip=self.proto_state_clip),
                            p=2,
                            dim=0,
                            eps=self.proto_eps,
                        )
                        for c in support_classes
                    ],
                    dim=0,
                )
                pairwise = torch.cdist(proto_stack, proto_stack, p=2)
                pairwise.fill_diagonal_(float("inf"))
                delta_k = float(pairwise.min().item()) if torch.isfinite(pairwise).any() else 0.0
            else:
                delta_k = 0.0
            eps_dir = float(np.max(direct_errs)) if direct_errs else 0.0
            eps_interp = float(np.max(interp_errs)) if interp_errs else 0.0
            eps_anchor = float(np.max(anchor_errs)) if anchor_errs else 0.0
            self.personalization_stats = {
                "epsilon_dir": eps_dir,
                "epsilon_interp": eps_interp,
                "epsilon_anchor": eps_anchor,
                "weighted_dir_mse": float(np.sum(w * (np.asarray(direct_errs) ** 2))),
                "weighted_interp_mse": float(np.sum(w * (np.asarray(interp_errs) ** 2))),
                "weighted_anchor_mse": float(np.sum(w * (np.asarray(anchor_errs) ** 2))),
                "rho_k": float(self.local_radius),
                "delta_k": delta_k,
                "C_dir": float(delta_k - 2.0 * self.local_radius - 2.0 * eps_dir),
                "C_interp": float(delta_k - 2.0 * self.local_radius - 2.0 * eps_interp),
                "C_anchor": float(delta_k - 2.0 * self.local_radius - 2.0 * eps_anchor),
            }
        else:
            self.personalization_stats = {}
        return anchor_bank, proto_weights

    def _apply_proto_logits(
        self,
        features: torch.Tensor,
        logits_cls: torch.Tensor,
        anchor_bank: Optional[torch.Tensor],
        proto_weights: Optional[torch.Tensor],
        local_class_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if anchor_bank is None or proto_weights is None:
            return logits_cls

        if local_class_mask is not None and local_class_mask.any():
            active_idx = torch.nonzero(
                local_class_mask.to(device=features.device), as_tuple=False
            ).view(-1)
        else:
            active_idx = torch.nonzero(
                proto_weights.to(device=features.device) > 0, as_tuple=False
            ).view(-1)

        if active_idx.numel() == 0:
            return logits_cls

        active_anchors = anchor_bank.index_select(0, active_idx).to(
            device=features.device, dtype=torch.float32
        )
        active_weights = proto_weights.index_select(0, active_idx).to(
            device=features.device, dtype=torch.float32
        )

        # r_{k,c}(x) = < \hat z_k(x), \hat q_{k,c} > / tau on c \in S_k
        feat_norm = F.normalize(features.to(dtype=torch.float32), p=2, dim=1, eps=self.proto_eps)
        anchor_norm = F.normalize(active_anchors, p=2, dim=1, eps=self.proto_eps)
        proto_scores = torch.matmul(feat_norm, anchor_norm.t()) / max(self.proto_temp, self.proto_eps)

        proto_term = torch.zeros_like(logits_cls)
        proto_term[:, active_idx] = (
            self.proto_logit_w
            * proto_scores.to(dtype=logits_cls.dtype)
            * active_weights.unsqueeze(0).to(dtype=logits_cls.dtype)
        )
        return logits_cls + proto_term

    def _proto_ce_loss(
        self,
        features: torch.Tensor,
        y: torch.Tensor,
        anchor_bank: Optional[torch.Tensor],
        local_class_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if anchor_bank is None or self.proto_cls_w <= 0 or not self.use_proto_ce_loss:
            return features.new_zeros(())
        if local_class_mask is not None and local_class_mask.any():
            active_idx = torch.nonzero(
                local_class_mask.to(device=features.device), as_tuple=False
            ).view(-1)
        else:
            active_idx = torch.arange(self.num_classes, device=features.device)

        if active_idx.numel() == 0:
            return features.new_zeros(())

        # L_proto = CE(([ <\hat z_k(x), \hat q_{k,c}> / tau ]_{c \in S_k}), y)
        active_anchors = anchor_bank.index_select(0, active_idx).to(
            device=features.device, dtype=torch.float32
        )
        feat_norm = F.normalize(features.to(dtype=torch.float32), p=2, dim=1, eps=self.proto_eps)
        anchor_norm = F.normalize(active_anchors, p=2, dim=1, eps=self.proto_eps)
        proto_logits = torch.matmul(feat_norm, anchor_norm.t()) / max(self.proto_temp, self.proto_eps)

        target_map = torch.full(
            (self.num_classes,),
            -1,
            device=features.device,
            dtype=torch.long,
        )
        target_map[active_idx] = torch.arange(active_idx.numel(), device=features.device)
        y_local = target_map[y.long()]
        valid = y_local >= 0
        if valid.sum().item() == 0:
            return features.new_zeros(())
        return F.cross_entropy(proto_logits[valid], y_local[valid])

    def _prepare_personalization(self, data_loader):
        self._build_local_prototypes(data_loader)
        self.local_radius = self._compute_local_radius(data_loader)
        self._estimate_affine_aligner(self.local_protos_cache)
        return (
            self._should_use_personalized(),
            *self._build_anchor_bank(),
            self._build_local_class_mask(),
        )

    def _run_local_epoch(
        self,
        trainloader,
        use_personalized: bool,
        anchor_bank,
        proto_weights,
        local_class_mask,
    ):
        self.model.train()
        for x, y in trainloader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))

            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                rep = self.model.base(x)
                logits_cls = self.model.head(rep)
                if use_personalized:
                    output = self._apply_proto_logits(
                        rep, logits_cls, anchor_bank, proto_weights, local_class_mask
                    )
                    proto_loss = self._proto_ce_loss(rep, y, anchor_bank, local_class_mask)
                else:
                    output = logits_cls
                    proto_loss = rep.new_zeros(())
                classification_loss = self.loss(output, y)
                loss = classification_loss + self.proto_cls_w * proto_loss

            self.optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                continue
            if self.amp_enabled:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.proto_grad_clip)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.proto_grad_clip)
                self.optimizer.step()

    def train(self):
        start_time = time.time()
        self.model.to(self.device)
        trainloader = self.load_train_data()

        use_personalized = self._should_use_personalized()

        warmup_epochs = self.local_epochs
        if use_personalized and self.local_epochs > 1:
            warmup_epochs = min(self.proto_warmup_epochs, self.local_epochs - 1)
        remain_epochs = max(self.local_epochs - warmup_epochs, 0)

        for _ in range(warmup_epochs):
            self._run_local_epoch(
                trainloader,
                use_personalized=False,
                anchor_bank=None,
                proto_weights=None,
                local_class_mask=None,
            )

        _, anchor_bank, proto_weights, local_class_mask = self._prepare_personalization(trainloader)

        if use_personalized and remain_epochs > 0 and anchor_bank is not None:
            for _ in range(remain_epochs):
                self._run_local_epoch(
                    trainloader,
                    use_personalized=True,
                    anchor_bank=anchor_bank,
                    proto_weights=proto_weights,
                    local_class_mask=local_class_mask,
                )

        self.protos = self.collect_protos(trainloader)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    @torch.no_grad()
    def collect_protos(self, data_loader=None):
        local_protos, _ = self._build_local_prototypes(data_loader)
        if len(local_protos) == 0:
            self.protos = {"shared": {}}
            return self.protos

        # Server-side aggregation keeps raw local prototypes.
        shared_protos = {
            int(c): self._clip_vec_norm(
                self._sanitize_vec(p, clip=self.proto_state_clip),
                self.proto_upload_norm,
            ).detach()
            for c, p in local_protos.items()
        }
        shared = {}
        for c, p in shared_protos.items():
            p_cpu = torch.nan_to_num(p.detach(), nan=0.0, posinf=0.0, neginf=0.0).cpu()
            shared[c] = p_cpu
        self.protos = {"shared": shared}
        return self.protos

    def set_protos(self, global_protos):
        clean = {}
        if global_protos is None:
            self.global_protos = {}
            return
        for c, p in global_protos.items():
            try:
                vec = self._sanitize_vec(p, clip=self.proto_state_clip)
            except Exception:
                continue
            if torch.isfinite(vec).all():
                clean[int(c)] = vec
        self.global_protos = clean

    def test_metrics(self, test_loader=None):
        if test_loader is None:
            test_loader = self.testloader

        trainloader = self.load_train_data()
        use_personalized, anchor_bank, proto_weights, local_class_mask = self._prepare_personalization(trainloader)
        self.model.eval()
        self.model.to(self.device)

        correct = 0
        total = 0
        y_prob_list = []
        y_true_list = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    rep = self.model.base(x)
                    logits_cls = self.model.head(rep)
                    logits = self._apply_proto_logits(
                        rep, logits_cls, anchor_bank, proto_weights, local_class_mask
                    )
                    probs = F.softmax(logits, dim=1)
                y_prob_list.append(probs.cpu().numpy())
                y_true_list.append(F.one_hot(y, num_classes=self.num_classes).cpu().numpy())
                total += y.size(0)
                correct += (torch.argmax(probs, dim=1) == y).sum().item()

        if len(y_prob_list) == 0:
            return 0, 0, np.array([]), np.array([])
        return correct, total, np.concatenate(y_prob_list, axis=0), np.concatenate(y_true_list, axis=0)

    def train_metrics(self):
        trainloader = self.load_train_data()
        use_personalized, anchor_bank, proto_weights, local_class_mask = self._prepare_personalization(trainloader)
        self.model.to(self.device, non_blocking=True)
        self.model.eval()

        train_correct = 0
        train_num = 0
        losses = 0.0
        with torch.no_grad():
            for x, y in trainloader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    rep = self.model.base(x)
                    logits_cls = self.model.head(rep)
                    output = self._apply_proto_logits(
                        rep, logits_cls, anchor_bank, proto_weights, local_class_mask
                    )
                    proto_loss = (
                        self._proto_ce_loss(rep, y, anchor_bank, local_class_mask)
                        if use_personalized else rep.new_zeros(())
                    )
                    classification_loss = self.loss(output, y)
                    loss = classification_loss + self.proto_cls_w * proto_loss
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
                train_correct += (torch.argmax(output, dim=1) == y).sum().item()
        return losses, train_num, train_correct
