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

    def forward(self, paf_pred, hm_pred, paf_gt, hm_gt, paf_mask):
        # 1. PAF Loss: 模型输出已经是 [-1, 1]（tanh在模型输出层）
        paf_diff = (paf_pred - paf_gt) * paf_mask

        # 全局 MSE，避免稀疏通道问题
        paf_loss = ((paf_diff ** 2).sum() / (paf_mask.sum() + 1e-6))

        # 2. Heatmap Loss: 模型输出已经是 [0, 1]（sigmoid在模型输出层）
        weight = torch.where(hm_gt > 0.5,
                             torch.tensor(8.0, device=hm_gt.device),
                             torch.where(hm_gt > 0.1,
                                         torch.tensor(3.0, device=hm_gt.device),
                                         torch.tensor(1.0, device=hm_gt.device)))
        hm_loss = ((hm_pred - hm_gt) ** 2 * weight).mean()

        # print(f"  paf_loss={paf_loss * 0.5:.4f}  hm_loss={hm_loss * 0.5:.4f}")
        # 平衡两个 Loss
        return paf_loss * 0.5 + hm_loss * 0.5
