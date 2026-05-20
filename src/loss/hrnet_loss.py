"""
@Author: chaos
@Date: 2026/4/29
@Version：V1.0 
@Description：
"""
import torch
import torch.nn as nn


class HRNetLoss(nn.Module):
    """MSE loss for PAF and heatmap prediction with PAF mask weighting.

    Same structure as OpenPoseLoss but for HRNet's single-stage output:
      loss = MSE((paf_pred - paf_gt) * mask) + MSE(hm_pred - hm_gt)
    """

    def __init__(self):
        super().__init__()
        # 预注册权重常量，避免每次 forward 重复创建 tensor
        self.register_buffer('weight_high', torch.tensor(8.0))
        self.register_buffer('weight_mid', torch.tensor(3.0))
        self.register_buffer('weight_low', torch.tensor(1.0))

    def forward(self, paf_pred, hm_pred, paf_gt, hm_gt, paf_mask):
        # 1. PAF Loss: 模型输出已经是 [-1, 1]（tanh在模型输出层）只在有效区域计算
        paf_diff = (paf_pred - paf_gt) * paf_mask
        valid_pixels = paf_mask.sum() + 1e-8
        paf_loss = (paf_diff ** 2).sum() / valid_pixels

        # 使用 mean() 与 heatmap loss 归一化方式一致
        # paf_loss = (paf_diff ** 2).mean()

        # 2. Heatmap Loss: 模型输出已经是 [0, 1]（sigmoid在模型输出层）
        weight = torch.where(hm_gt > 0.5,
                             self.weight_high,
                             torch.where(hm_gt > 0.1,
                                         self.weight_mid,
                                         self.weight_low))

        # 对左右脚踝关键点（索引15和16）增加额外权重
        ankle_indices = [15, 16]  # left_ankle, right_ankle
        ankle_weight_boost = torch.ones_like(weight)
        for idx in ankle_indices:
            if idx < hm_gt.shape[1]:
                ankle_weight_boost[:, idx, :, :] *= 2.0

        weight = weight * ankle_weight_boost

        hm_loss = ((hm_pred - hm_gt) ** 2 * weight).mean()

        # 平衡两个 Loss
        return paf_loss * 0.7 + hm_loss * 0.3
