from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from flcore.trainmodel.alexnet import alexnet
from flcore.trainmodel.mobilenet_v2 import mobilenet_v2
from flcore.trainmodel.mobilenet_v3 import mobilenet_v3_ultralite
from flcore.trainmodel.models import (
    AmazonMLP,
    Digit5CNN,
    FashionCNNModel,
    FedAvgCNN,
    HARCNN,
)


class FedSTARModel(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        feature_dim: int,
        num_classes: int,
        shared_dim: int = 128,
        private_dim: int = 128,
        use_private_branch: bool = True,
        shared_classifier_scale: float = 16.0,
        backbone_name: str = "unknown",
    ):
        super().__init__()
        self.base = base
        self.feature_dim = int(feature_dim)
        self.shared_dim = int(shared_dim)
        self.private_dim = int(private_dim) if use_private_branch else 0
        self.use_private_branch = bool(use_private_branch)
        self.shared_classifier_scale = float(shared_classifier_scale)
        self.backbone_name = str(backbone_name)

        self.shared_projector = nn.Linear(self.feature_dim, self.shared_dim)
        self.private_projector = (
            nn.Linear(self.feature_dim, self.private_dim)
            if self.use_private_branch
            else None
        )
        self.shared_head = nn.Linear(self.shared_dim, num_classes, bias=False)
        self.private_head = (
            nn.Linear(self.private_dim, num_classes)
            if self.use_private_branch
            else None
        )
        if self.private_head is not None:
            nn.init.zeros_(self.private_head.weight)
            nn.init.zeros_(self.private_head.bias)

    def extract_backbone(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.base(x)
        if isinstance(feat, tuple):
            feat = feat[-1]
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)
        return feat

    def project(self, feat: torch.Tensor):
        z_s = self.shared_projector(feat)
        if self.use_private_branch and self.private_projector is not None:
            z_p = self.private_projector(feat)
        else:
            z_p = feat.new_zeros((feat.size(0), 0))
        return z_s, z_p

    def classify(self, z_s: torch.Tensor, z_p: torch.Tensor):
        z_s_norm = F.normalize(z_s, p=2, dim=1)
        shared_weight = F.normalize(self.shared_head.weight, p=2, dim=1)
        shared_logits = self.shared_classifier_scale * (z_s_norm @ shared_weight.T)
        if self.use_private_branch and self.private_head is not None and z_p.numel() > 0:
            private_logits = self.private_head(z_p)
        else:
            private_logits = shared_logits.new_zeros(shared_logits.shape)
        logits = shared_logits + private_logits
        return logits, shared_logits, private_logits

    def extract_embeddings(self, x: torch.Tensor):
        feat = self.extract_backbone(x)
        z_s, z_p = self.project(feat)
        return feat, z_s, z_p

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat, z_s, z_p = self.extract_embeddings(x)
        logits, shared_logits, private_logits = self.classify(z_s, z_p)
        if return_features:
            return logits, feat, z_s, z_p, shared_logits, private_logits
        return logits


class _FashionBackbone(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def parse_client_model_names(client_models: str, num_clients: int, default_model: str) -> List[str]:
    names = []
    if client_models:
        names = [name.strip() for name in str(client_models).split(",") if name.strip()]
    if len(names) == 0:
        names = [str(default_model)]
    if len(names) == 1:
        return names * int(num_clients)
    if len(names) == int(num_clients):
        return names
    return [names[idx % len(names)] for idx in range(int(num_clients))]


def _replace_linear_head_with_identity(model: nn.Module, attr_name: str = "fc"):
    head = getattr(model, attr_name, None)
    if head is None:
        raise NotImplementedError(f"Model {type(model).__name__} does not define `{attr_name}`.")

    if isinstance(head, nn.Linear):
        feature_dim = int(head.in_features)
        setattr(model, attr_name, nn.Identity())
        return model, feature_dim

    if isinstance(head, nn.Sequential):
        modules = list(head.children())
        last_linear_idx = -1
        last_linear = None
        for idx in range(len(modules) - 1, -1, -1):
            if isinstance(modules[idx], nn.Linear):
                last_linear_idx = idx
                last_linear = modules[idx]
                break
        if last_linear_idx < 0 or last_linear_idx != len(modules) - 1:
            raise NotImplementedError(
                f"Unsupported sequential head layout for {type(model).__name__}.{attr_name}"
            )
        modules[last_linear_idx] = nn.Identity()
        setattr(model, attr_name, nn.Sequential(*modules))
        return model, int(last_linear.in_features)

    raise NotImplementedError(
        f"Unsupported classifier type `{type(head).__name__}` for {type(model).__name__}.{attr_name}"
    )


def _build_raw_model(model_name: str, dataset: str, num_classes: int) -> nn.Module:
    if model_name == "fmnist":
        return FashionCNNModel(num_classes=num_classes)

    if model_name == "cnn":
        if "mnist" in dataset:
            return FedAvgCNN(in_features=1, num_classes=num_classes, dim=1024)
        if "Cifar10" in dataset:
            return FedAvgCNN(in_features=3, num_classes=num_classes, dim=1600)
        if "Tiny-imagenet" in dataset:
            return FedAvgCNN(in_features=3, num_classes=num_classes, dim=10816)
        if "omniglot" in dataset:
            return FedAvgCNN(in_features=1, num_classes=num_classes, dim=33856)
        if "Digit5" in dataset:
            return Digit5CNN()
        if "office" in dataset:
            return FedAvgCNN(in_features=3, num_classes=num_classes, dim=379456)
        return FedAvgCNN(in_features=3, num_classes=num_classes, dim=1600)

    if model_name == "resnet":
        import torchvision
        return torchvision.models.resnet18(pretrained=False, num_classes=num_classes)

    if model_name == "alexnet":
        return alexnet(pretrained=False, num_classes=num_classes)

    if model_name == "googlenet":
        import torchvision
        return torchvision.models.googlenet(
            pretrained=False,
            aux_logits=False,
            num_classes=num_classes,
        )

    if model_name == "mobilenet_v2":
        return mobilenet_v2(pretrained=False, num_classes=num_classes)

    if model_name == "mobilenet_v3":
        return mobilenet_v3_ultralite(pretrained=False, num_classes=num_classes)

    if model_name == "Digit5":
        return Digit5CNN()

    if model_name == "AmazonMLP":
        return AmazonMLP()

    if model_name == "harcnn":
        if dataset == "har":
            return HARCNN(
                9,
                dim_hidden=1664,
                num_classes=num_classes,
                conv_kernel_size=(1, 9),
                pool_kernel_size=(1, 2),
            )
        if dataset == "pamap":
            return HARCNN(
                9,
                dim_hidden=3712,
                num_classes=num_classes,
                conv_kernel_size=(1, 9),
                pool_kernel_size=(1, 2),
            )
        raise NotImplementedError(f"`harcnn` is only supported for har/pamap, got `{dataset}`.")

    raise NotImplementedError(
        f"FedSTAR currently supports backbones in "
        f"[fmnist, cnn, resnet, alexnet, googlenet, mobilenet_v2, mobilenet_v3, Digit5, AmazonMLP, harcnn], "
        f"got `{model_name}`."
    )


def build_fedstar_model(
    model_name: str,
    dataset: str,
    num_classes: int,
    shared_dim: int = 128,
    private_dim: int = 128,
    use_private_branch: bool = True,
    shared_classifier_scale: float = 16.0,
) -> FedSTARModel:
    raw_model = _build_raw_model(model_name=model_name, dataset=dataset, num_classes=num_classes)

    if model_name == "fmnist":
        base = _FashionBackbone(raw_model.backbone)
        feature_dim = int(raw_model.fc.in_features)
    else:
        base, feature_dim = _replace_linear_head_with_identity(raw_model, attr_name="fc")

    return FedSTARModel(
        base=base,
        feature_dim=feature_dim,
        num_classes=num_classes,
        shared_dim=shared_dim,
        private_dim=private_dim,
        use_private_branch=use_private_branch,
        shared_classifier_scale=shared_classifier_scale,
        backbone_name=model_name,
    )
