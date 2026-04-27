"""Training script for OpenPose body pose model on COCO keypoints.

Dataset expected format (COCO-style):
  - Images in <data_dir>/images/<split>/
  - Annotations in <data_dir>/annotations/person_keypoints_<split>.json

Usage:
  python train.py --data_dir /path/to/coco --epochs 100 --batch_size 8
"""

import argparse
import os
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils.CustomDataSet import CustomDataSet

# Enable TF32 for faster matrix operations on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Enable benchmark mode for optimal convolution algorithms
torch.backends.cudnn.benchmark = True

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from src.config import (
    NUM_JOINTS, NUM_LIMBS, DEBUG, DEBUG_COMPARE_GT_DIR
)
from src.model import BodyPoseTrainModel


# --- Loss ---------------------------------------------------------------------

class OpenPoseLoss(nn.Module):
    """MSE loss summed over all 6 stages, weighted by PAF mask."""

    def forward(self, paf_stages, hm_stages, paf_gt, hm_gt, paf_mask):
        loss = 0.0
        for paf_pred, hm_pred in zip(paf_stages, hm_stages):
            paf_diff = (paf_pred - paf_gt) * paf_mask
            # 计算有效像素数量，避免被大量 0 值稀释
            # valid_pixels = paf_mask.sum() + 1e-6
            # loss += (paf_diff ** 2).sum() / valid_pixels
            loss += (paf_diff ** 2).mean()
            loss += ((hm_pred - hm_gt) ** 2).mean()
            print(f"[DEBUG] hm loss: {((hm_pred - hm_gt) ** 2).mean():.4f}, paf loss: {(paf_diff ** 2).mean():.4f} ")
        return loss


def compare_gt_and_pred(model, imgs, paf_gt, hm_gt, paf_mask, paf_stages, hm_stages):
    img_np = imgs[0].cpu().numpy()
    img_vis = (img_np.transpose(1, 2, 0) + 0.5) * 255
    img_vis = img_vis.astype(np.uint8)

    hm_np = hm_gt[0].cpu().numpy()
    paf_np = paf_gt[0].cpu().numpy()
    hm_vis_data = hm_np.transpose(1, 2, 0)
    paf_vis_data = paf_np.transpose(1, 2, 0)

    os.makedirs(DEBUG_COMPARE_GT_DIR, exist_ok=True)

    # 1. 保存 GT Heatmap 可视化
    for part in range(min(NUM_JOINTS, hm_vis_data.shape[2])):
        print(f"[DEBUG] GT HM Channel {part} max value: {hm_vis_data[:, :, part].max():.4f}")
        hm_single = hm_vis_data[:, :, part]
        hm_norm = (hm_single * 255).astype(np.uint8)
        hm_colored = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        hm_colored = cv2.resize(hm_colored, (img_vis.shape[1], img_vis.shape[0]))
        blend = cv2.addWeighted(img_vis, 0.5, hm_colored, 0.5, 0)
        cv2.imwrite(os.path.join(DEBUG_COMPARE_GT_DIR, f'GT_hm_{part:02d}.jpg'), blend)

    # 2. 保存 GT PAF 可视化 (取每对通道的幅值)
    for limb in range(NUM_LIMBS):
        paf_x = paf_vis_data[:, :, limb * 2]
        paf_y = paf_vis_data[:, :, limb * 2 + 1]
        paf_mag = np.sqrt(paf_x ** 2 + paf_y ** 2)
        paf_norm = (np.clip(paf_mag, 0, 1) * 255).astype(np.uint8)
        paf_colored = cv2.applyColorMap(paf_norm, cv2.COLORMAP_HOT)
        paf_colored = cv2.resize(paf_colored, (img_vis.shape[1], img_vis.shape[0]))
        blend = cv2.addWeighted(img_vis, 0.6, paf_colored, 0.4, 0)
        cv2.imwrite(os.path.join(DEBUG_COMPARE_GT_DIR, f'GT_paf_{limb:02d}.jpg'), blend)

    # 3. 保存 GT PAF Mask (新增)
    mask_vis_data = paf_mask[0].cpu().numpy().transpose(1, 2, 0)
    for limb in range(NUM_LIMBS):
        # Mask 是 0/1 二值图，直接映射
        mask_single = mask_vis_data[:, :, limb * 2]  # x 和 y 的 mask 是一样的
        mask_uint8 = (mask_single * 255).astype(np.uint8)
        mask_colored = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_BONE)
        mask_colored = cv2.resize(mask_colored, (img_vis.shape[1], img_vis.shape[0]))
        # 叠加显示：白色区域表示模型应该学习的肢体范围
        blend = cv2.addWeighted(img_vis, 0.7, mask_colored, 0.3, 0)
        cv2.imwrite(os.path.join(DEBUG_COMPARE_GT_DIR, f'GT_mask_{limb:02d}.jpg'), blend)

    hm_pred = hm_stages[-1][0].detach().cpu().numpy()
    paf_pred = paf_stages[-1][0].detach().cpu().numpy()

    hm_pred_data = hm_pred.transpose(1, 2, 0)
    paf_pred_data = paf_pred.transpose(1, 2, 0)

    # 1. 保存 PRED Heatmap 可视化
    for part in range(min(NUM_JOINTS, hm_pred_data.shape[2])):
        print(f"[DEBUG] PRED HM Channel {part} max value: {hm_pred_data[:, :, part].max():.4f}")
        hm_single = hm_pred_data[:, :, part]
        hm_norm = (hm_single * 255).astype(np.uint8)
        hm_colored = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        hm_colored = cv2.resize(hm_colored, (img_vis.shape[1], img_vis.shape[0]))
        blend = cv2.addWeighted(img_vis, 0.5, hm_colored, 0.5, 0)
        cv2.imwrite(os.path.join(DEBUG_COMPARE_GT_DIR, f'PRED_hm_{part:02d}.jpg'), blend)

    # 2. 保存 PRED PAF 可视化
    for limb in range(NUM_LIMBS):
        paf_x = paf_pred_data[:, :, limb * 2]
        paf_y = paf_pred_data[:, :, limb * 2 + 1]
        paf_mag = np.sqrt(paf_x ** 2 + paf_y ** 2)
        paf_norm = (np.clip(paf_mag, 0, 1) * 255).astype(np.uint8)
        paf_colored = cv2.applyColorMap(paf_norm, cv2.COLORMAP_HOT)
        paf_colored = cv2.resize(paf_colored, (img_vis.shape[1], img_vis.shape[0]))
        blend = cv2.addWeighted(img_vis, 0.6, paf_colored, 0.4, 0)
        cv2.imwrite(os.path.join(DEBUG_COMPARE_GT_DIR, f'PRED_paf_{limb:02d}.jpg'), blend)

    print(f"Comparison images saved to: {DEBUG_COMPARE_GT_DIR}")


# --- Training Loop ------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BodyPoseTrainModel().to(device)

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state)
        print(f"Resumed from {args.resume}")

    train_ds = CustomDataSet(args.data_dir, split="train2017", preprocessed_dir=args.preprocessed_dir,
                             load_to_memory=args.load_to_memory)
    val_ds = CustomDataSet(args.data_dir, split="val2017", preprocessed_dir=args.preprocessed_dir,
                           load_to_memory=args.load_to_memory)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True,
                              prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=True,
                            prefetch_factor=4, persistent_workers=True)

    criterion = OpenPoseLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(args.epochs * 0.6), int(args.epochs * 0.85)], gamma=0.1)
    # 2、Warmup + 余弦退火
    # warmup_epochs = 5
    # def warmup_cosine(epoch):
    #     if epoch < warmup_epochs:
    #         return (epoch + 1) / warmup_epochs
    #     return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))
    #
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine)
    # 3、三阶段学习率调度：Warmup → 快速收敛 → 精细微调
    # warmup_epochs = 5
    # fast_converge_end = int(args.epochs * 0.6)
    #
    # def three_stage_lr(epoch):
    #     """
    #     阶段1 (0-5): Warmup - 从0线性增长到1.0
    #     阶段2 (5-60): 快速收敛 - 保持高学习率，余弦衰减到0.1
    #     阶段3 (60-100): 精细微调 - 低学习率，缓慢衰减到0.01
    #     """
    #     if epoch < warmup_epochs:
    #         # Warmup阶段：线性增长
    #         return (epoch + 1) / warmup_epochs
    #     elif epoch < fast_converge_end:
    #         # 快速收敛阶段：余弦衰减到高学习率的10%
    #         progress = (epoch - warmup_epochs) / (fast_converge_end - warmup_epochs)
    #         return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))
    #     else:
    #         # 精细微调阶段：进一步衰减到1%
    #         progress = (epoch - fast_converge_end) / (args.epochs - fast_converge_end)
    #         return 0.01 + 0.09 * 0.5 * (1 + np.cos(np.pi * progress))
    #
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=three_stage_lr)

    # 充分利用 Tensor Cores，启用FP16 混合精度
    scaler = torch.amp.GradScaler('cuda', enabled=True, init_scale=2 ** 16)
    writer = None
    if SummaryWriter is not None:
        try:
            writer = SummaryWriter(log_dir=args.log_dir)
        except Exception as e:
            print(f"Warning: Failed to initialize TensorBoard: {e}")

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        # ── train ──
        model.train()
        train_loss = 0.0
        # 清零梯度
        optimizer.zero_grad(set_to_none=True)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch {epoch} start")
        last_step_finished_time = time.time()
        for step, (imgs, paf_gt, hm_gt, paf_mask) in enumerate(train_loader):
            data_time = time.time()
            # Use non_blocking for async transfer
            imgs = imgs.to(device, non_blocking=True)
            paf_gt = paf_gt.to(device)
            hm_gt = hm_gt.to(device)
            paf_mask = paf_mask.to(device)

            with torch.amp.autocast('cuda'):
                paf_stages, hm_stages = model(imgs)
                loss = criterion(paf_stages, hm_stages, paf_gt, hm_gt, paf_mask)
                loss = loss / args.accumulation_steps

            # 输出Ground Truth可视化和模型输出可视化
            if DEBUG and step == 0:
                compare_gt_and_pred(model, imgs, paf_gt, hm_gt, paf_mask, paf_stages, hm_stages)

            forward_time = time.time()
            scaler.scale(loss).backward()

            # Gradient accumulation: only update after N steps
            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            backward_time = time.time()
            train_loss += loss.item() * args.accumulation_steps

            if DEBUG and step < 10:
                step_total = backward_time - data_time
                print(f"Step {step + 1}: Data={data_time - last_step_finished_time:.3f}s | "
                      f"Forward={forward_time - data_time:.3f}s | "
                      f"Backward={backward_time - forward_time:.3f}s | "
                      f"Total={step_total:.3f}s")

            if (step + 1) % 50 == 0:
                elapsed = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                progress = (step + 1) / len(train_loader) * 100
                eta_seconds = (time.time() - epoch_start) / (step + 1) * (len(train_loader) - step - 1)
                eta_minutes = eta_seconds / 60
                print(f"{elapsed}  Epoch {epoch} [{progress:.1f}%] step {step + 1}/{len(train_loader)}  "
                      f"loss={loss.item() * args.accumulation_steps:.4f}  ETA={eta_minutes:.1f}min")

            last_step_finished_time = time.time()

        avg_train = train_loss / len(train_loader)
        if writer is not None:
            try:
                writer.add_scalar("loss/train", avg_train, epoch)
            except Exception:
                pass

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, paf_gt, hm_gt, paf_mask in val_loader:
                imgs = imgs.to(device)
                paf_gt = paf_gt.to(device)
                hm_gt = hm_gt.to(device)
                paf_mask = paf_mask.to(device)

                with torch.amp.autocast('cuda'):
                    paf_stages, hm_stages = model(imgs)
                    val_loss += criterion(paf_stages, hm_stages, paf_gt, hm_gt, paf_mask).item()

        avg_val = val_loss / len(val_loader)
        if writer is not None:
            try:
                writer.add_scalar("loss/val", avg_val, epoch)
            except Exception:
                pass

        scheduler.step()

        print(
            f"Epoch {epoch}/{args.epochs}  train={avg_train:.4f}  val={avg_val:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f"best_epoch{epoch:04d}_loss{best_val:.4f}.pth"))

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f"epoch{epoch:04d}_loss{avg_val:.4f}.pth"))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    if writer is not None:
        try:
            writer.close()
        except Exception:
            pass
    print(f"Training complete. Best val loss: {best_val}")

# --- Entry Point --------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OpenPose body pose model")
    parser.add_argument("--data_dir", default="./data")
    # parser.add_argument("--preprocessed_dir", default="./preprocessed")
    parser.add_argument("--preprocessed_dir", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    # 梯度累积模拟更大 batch_size
    parser.add_argument("--accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps (effective batch_size = batch_size * accumulation_steps)")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--load_to_memory", action="store_true",
                        help="Load all data into RAM for fastest training",
                        default=False)
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--log_dir", default="runs")
    parser.add_argument("--save_every", type=int, default=1)
    # parser.add_argument("--resume", default="./checkpoints/epoch0004_loss0.1210.pth")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train(args)
