import torch
import torch.nn as nn
import torch.nn.functional as F

# class SupConLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, features, labels):
#         # features: [B, n_views, D]
#         device = features.device
#         B, n_views, D = features.shape

#         # Normalize
#         features = F.normalize(features, dim=2)

#         contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)  # [B*n_views, D]
#         labels = labels.contiguous().view(-1, 1)
#         mask = torch.eq(labels, labels.T).float().to(device)  # [B, B]

#         contrast_count = n_views
#         anchor_feature = contrast_features
#         anchor_count = contrast_count

#         anchor_dot_contrast = torch.div(
#             torch.matmul(anchor_feature, contrast_features.T),
#             self.temperature
#         )  # [B*n_views, B*n_views]

#         # mask out self-comparisons
#         logits_mask = torch.scatter(
#             torch.ones_like(anchor_dot_contrast),
#             1,
#             torch.arange(B * n_views).view(-1, 1).to(device),
#             0
#         )
#         mask = mask.repeat(anchor_count, contrast_count)
#         mask = mask * logits_mask

#         # compute log prob
#         exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
#         log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

#         mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

#         # loss
#         loss = -mean_log_prob_pos
#         loss = loss.mean()
#         return loss

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020)
    features: [B, n_views, D]
    labels:   [B]
    """
    def __init__(self, temperature: float = 0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        B, n_views, D = features.shape
        features = F.normalize(features, dim=2)

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = n_views
        anchor_feature = contrast_features
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_features.T), self.temperature
        )

        logits_mask = torch.ones_like(anchor_dot_contrast)
        logits_mask.fill_diagonal_(0)
        mask = mask.repeat(contrast_count, contrast_count) * logits_mask

        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        return -mean_log_prob_pos.mean()