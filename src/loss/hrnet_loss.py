
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
        self.alpha_neg = 0.25   # 背景基础权重
        self.gamma_neg = 2.0    # 聚焦因子

        # 2. 不确定性加权参数 (Learnable uncertainty)
        # 初始化为 0，即初始权重为 e^0 = 1.0，让模型在训练中自适应调整
        self.log_sigma_paf = nn.Parameter(torch.zeros(1))
        self.log_sigma_hm = nn.Parameter(torch.zeros(1))

    def forward(self, paf_pred, hm_pred, paf_gt, hm_gt, paf_mask):
        device = paf_pred.device
        
        # ─ 1. PAF Loss: 逐通道计算 + 增强背景抑制 ──
        num_limbs = paf_pred.shape[1] // 2
        paf_loss_per_limb = []
        background_weight = 0.5  # 增强背景抑制（从 0.1 提升到 0.5）

        for limb_idx in range(num_limbs):
            # 提取当前肢体通道
            px, py = paf_pred[:, limb_idx * 2], paf_pred[:, limb_idx * 2 + 1]
            gx, gy = paf_gt[:, limb_idx * 2], paf_gt[:, limb_idx * 2 + 1]
            mx, my = paf_mask[:, limb_idx * 2], paf_mask[:, limb_idx * 2 + 1]

            # 正样本 Loss：肢体内部拟合方向
            # 使用 Sum / Count 归一化，确保稀疏肢体和密集肢体贡献一致
            pos_x = ((px - gx) * mx).pow(2).sum() / (mx.sum() + 1e-8)
            pos_y = ((py - gy) * my).pow(2).sum() / (my.sum() + 1e-8)
            loss_pos = (pos_x + pos_y) / 2

            # 负样本 Loss：背景区域强制归零（解决 PAF 退化为 Heatmap 的问题）
            neg_x = (px * (1 - mx)).pow(2).mean()
            neg_y = (py * (1 - my)).pow(2).mean()
            loss_neg = (neg_x + neg_y) / 2

            paf_loss_per_limb.append(loss_pos + background_weight * loss_neg)

        # 对所有肢体 Loss 取平均，消除"躯干主导"现象
        paf_loss = torch.stack(paf_loss_per_limb).mean()

        # ── 2. Heatmap Loss: Focal-style 自适应抑制假阳性 ──
        # 定义正样本掩码 (hm_gt > 0.1 视为关键点影响区)
        mask_positive = (hm_gt > 0.1).float()
        mask_negative = 1.0 - mask_positive

        # 正样本权重分配
        weight_pos = torch.where(hm_gt > 0.5,
                                 self.weight_high,
                                 self.weight_mid)

        # 负样本权重（Focal-style 动态权重）
        # 原理：预测值越接近 0（正确），权重越小
        #       预测值越高（假阳性），权重越大
        hm_pred_clamped = hm_pred.clamp(0, 1)
        neg_weight = self.alpha_neg * (hm_pred_clamped / (hm_pred_clamped + 1e-8)).pow(self.gamma_neg)

        # 合并正负样本权重
        final_weight = torch.where(mask_positive > 0.5, weight_pos, neg_weight)

        # 计算 Heatmap 损失（使用 mean 保持数值稳定）
        hm_loss = ((hm_pred - hm_gt).pow(2) * final_weight).mean()

        # ─ 3. 自动平衡 (Uncertainty Weighting) ──
        # 模型会根据任务难度自动调整 paf_loss 和 hm_loss 的权重比例
        total_loss = (paf_loss * torch.exp(-self.log_sigma_paf) + self.log_sigma_paf +
                      hm_loss * torch.exp(-self.log_sigma_hm) + self.log_sigma_hm)
        
        return total_loss