"""
@Author: chaos
@Date: 2026/4/29
@Version：V1.0 
@Description：
"""
import torch
import torch.nn as nn


class HRNetLoss(nn.Module):
    """Optimized Loss for HRNet body pose.

    Features:
      1. PAF: Per-limb averaging (solves limb imbalance) + 增强背景抑制.
      2. Heatmap: Focal-style weighting (自适应抑制假阳性).
      3. Balancing: Uncertainty weighting (auto-adjusts loss coefficients).
    """

    def __init__(self):
        super().__init__()
        # 1. Heatmap 权重配置 (Focal-style)
        # 正样本高权重，负样本动态权重（假阳性越严重惩罚越大）
        self.register_buffer('weight_high', torch.tensor(32.0))
        self.register_buffer('weight_mid', torch.tensor(8.0))

        # Focal Loss 参数
        self.alpha_neg = 0.25  # 背景基础权重
        self.gamma_neg = 2.0  # 聚焦因子

        # 2. 不确定性加权参数 (Learnable uncertainty)
        # 初始权重：PAF更难学，给予更高权重
        # exp(-0.5) ≈ 0.6, exp(0) = 1.0
        self.log_sigma_paf = nn.Parameter(torch.tensor(-0.5))
        self.log_sigma_hm = nn.Parameter(torch.tensor(0.0))

    def forward(self, paf_pred, hm_pred, paf_gt, hm_gt, paf_mask):
        # 1. PAF Loss (向量化)
        B, C, H, W = paf_pred.shape
        num_limbs = C // 2
        paf_pred_r = paf_pred.reshape(B, num_limbs, 2, H, W)
        paf_gt_r = paf_gt.reshape(B, num_limbs, 2, H, W)
        paf_mask_r = paf_mask.reshape(B, num_limbs, 2, H, W)

        # 正样本
        pos_diff = (paf_pred_r - paf_gt_r) * paf_mask_r
        pos_loss = (pos_diff.pow(2).sum(dim=[0, 3, 4]) / (paf_mask_r.sum(dim=[0, 3, 4]) + 1.0)).mean(dim=1)

        # 负样本
        neg_mask_r = 1.0 - paf_mask_r
        neg_diff = paf_pred_r * neg_mask_r
        neg_loss = (neg_diff.pow(2).sum(dim=[0, 3, 4]) / (neg_mask_r.sum(dim=[0, 3, 4]) + 1.0)).mean(dim=1)

        paf_loss = (pos_loss + 0.1 * neg_loss).mean()

        # 2. Heatmap Loss
        weight_pos = torch.where(hm_gt > 0.5, self.weight_high, self.weight_mid)
        neg_weight = self.alpha_neg * hm_pred.pow(self.gamma_neg)
        final_weight = torch.where(hm_gt > 0.1, weight_pos, neg_weight)
        hm_loss = ((hm_pred - hm_gt).pow(2) * final_weight).mean()

        # 3. Uncertainty Weighting (公式：0.5*exp(-2*s))
        total_loss = (0.5 * paf_loss * torch.exp(-2 * self.log_sigma_paf) + self.log_sigma_paf +
                      0.5 * hm_loss * torch.exp(-2 * self.log_sigma_hm) + self.log_sigma_hm)

        return total_loss
