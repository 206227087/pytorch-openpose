"""Training script for HRNet body pose model on COCO keypoints.

Follows the same structure as train.py (OpenPose training) but adapted for
HRNet's dual-branch (PAF + heatmap) architecture for multi-person pose
estimation.

Key differences from OpenPose training:
  - HRNet outputs 32-ch PAF + 17-ch heatmap (COCO 16 limbs, 17 keypoints)
  - Uses MSE loss on both PAF (with mask weighting) and heatmap
  - Uses Adam optimizer with warmup + cosine annealing (standard for HRNet)
  - ImageNet normalization instead of [-0.5, 0.5]
  - Input size 256x256 (vs OpenPose's 368x368)

Dataset expected format (COCO-style):
  - Images in <data_dir>/images/<split>/
  - Annotations in <data_dir>/annotations/person_keypoints_<split>.json

Usage:
  python train_hrnet.py --data_dir /path/to/coco --epochs 210 --batch_size 32
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

from src.utils.HRNetCocoDataset import HRNetCocoDataset

# Enable TF32 for faster matrix operations on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Enable benchmark mode for optimal convolution algorithms
torch.backends.cudnn.benchmark = True

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from src.model import HRNet


# --- Loss ---------------------------------------------------------------------

class HRNetLoss(nn.Module):
    """MSE loss for PAF and heatmap prediction with PAF mask weighting.

    Same structure as OpenPoseLoss but for HRNet's single-stage output:
      loss = MSE((paf_pred - paf_gt) * mask) + MSE(hm_pred - hm_gt)
    """

    # def forward(self, paf_pred, hm_pred, paf_gt, hm_gt, paf_mask):
    #     # paf_diff = (paf_pred - paf_gt) * paf_mask
    #     # paf_loss = (paf_diff ** 2).mean()
    #     # hm_loss = ((hm_pred - hm_gt) ** 2).mean()
    #     paf_diff = (paf_pred - paf_gt) * paf_mask
    #     valid_pixels = paf_mask.sum() + 1e-6
    #     paf_loss = (paf_diff ** 2).sum() / valid_pixels
    #
    #     # 给 Heatmap 加权重，强迫模型学习关键点
    #     hm_loss = ((hm_pred - hm_gt) ** 2).mean() * 5.0
    #     # print(f"  paf_loss={paf_loss:.4f}  hm_loss={hm_loss:.4f}")
    #     return paf_loss + hm_loss
    # def forward(self, paf_pred, hm_pred, paf_gt, hm_gt, paf_mask):
    #     # 1. PAF Loss: 只在有肢体的区域计算
    #     paf_diff = (paf_pred - paf_gt) * paf_mask
    #     valid_pixels = paf_mask.sum() + 1e-6
    #     paf_loss = (paf_diff ** 2).sum() / valid_pixels
    #
    #     # 2. Heatmap Loss: 关键点区域权重加大 10 倍
    #     # 只有当 GT > 0.1 时（即关键点高斯分布的中心区域），才给高权重
    #     # 这样模型如果输出 0，Loss 会很大，强迫它输出高值
    #     weight = torch.where(hm_gt > 0.1, torch.tensor(10.0, device=hm_gt.device),
    #                          torch.tensor(1.0, device=hm_gt.device))
    #     hm_loss = ((hm_pred - hm_gt) ** 2 * weight).mean()
    #
    #     return paf_loss + hm_loss
    def forward(self, paf_pred, hm_pred, paf_gt, hm_gt, paf_mask):
        # 1. PAF Loss: 改进数值稳定性
        # 方法 A：逐通道归一化，避免全局 mask 导致的稀疏问题
        # 对 PAF 输出进行裁剪，避免数值爆炸
        paf_pred_clamped = torch.clamp(paf_pred, -1.0, 1.0)
        paf_diff = (paf_pred_clamped - paf_gt) * paf_mask

        # 对每个通道单独计算 mean，然后取平均
        # paf_diff shape: (B, 32, H, W)
        # paf_mask shape: (B, 32, H, W)
        channel_mask_sum = paf_mask.sum(dim=[0, 2, 3], keepdim=True) + 1e-6  # (1, 32, 1, 1)
        paf_loss = ((paf_diff ** 2).sum(dim=[0, 2, 3], keepdim=True) / channel_mask_sum).mean()

        # 方法 B（备选）：直接用 mean，但限制 PAF 输出的范围
        # paf_loss = ((paf_pred - paf_gt) ** 2 * paf_mask).mean()

        # 2. Heatmap Loss
        hm_pred_sigmoid = torch.sigmoid(hm_pred)
        weight = torch.where(hm_gt > 0.1, torch.tensor(10.0, device=hm_gt.device),
                             torch.tensor(1.0, device=hm_gt.device))
        hm_loss = ((hm_pred_sigmoid - hm_gt) ** 2 * weight).mean()

        # print(f"  paf_loss={paf_loss * 0.5:.4f}  hm_loss={hm_loss * 0.5:.4f}")
        # 平衡两个 Loss
        return paf_loss * 0.5 + hm_loss * 0.5


# --- Debug Visualization ------------------------------------------------------

DEBUG_OUTPUT_DIR = 'output/hrnet_train'


def save_heatmap_comparison(img, hm_pred, hm_gt, paf_pred, paf_gt, epoch, save_dir):
    """Save comparison of predicted vs GT heatmaps and PAF.

    Args:
        img: (3, H, W) normalized image tensor.
        hm_pred: (K, h, w) predicted heatmap numpy array.
        hm_gt: (K, h, w) ground truth heatmap numpy array.
        paf_pred: (C, h, w) predicted PAF numpy array.
        paf_gt: (C, h, w) ground truth PAF numpy array.
        epoch: current epoch number.
        save_dir: directory to save images.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Denormalize image (ImageNet)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img.cpu().numpy().transpose(1, 2, 0)
    img_vis = ((img_np * std + mean) * 255).clip(0, 255).astype(np.uint8)

    # Visualize heatmap joints
    for j in range(hm_pred.shape[0]):
        gt_hm = hm_gt[j].astype(np.float32)
        gt_hm = cv2.resize(gt_hm, (img_vis.shape[1], img_vis.shape[0]))
        gt_hm = cv2.normalize(gt_hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        gt_hm = cv2.applyColorMap(gt_hm, cv2.COLORMAP_JET)
        gt_blend = cv2.addWeighted(img_vis, 0.5, gt_hm, 0.5, 0)

        pred_hm = hm_pred[j].astype(np.float32)
        pred_hm = cv2.resize(pred_hm, (img_vis.shape[1], img_vis.shape[0]))
        pred_hm = cv2.normalize(pred_hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        pred_hm = cv2.applyColorMap(pred_hm, cv2.COLORMAP_JET)
        pred_blend = cv2.addWeighted(img_vis, 0.5, pred_hm, 0.5, 0)

        comparison = np.concatenate([gt_blend, pred_blend], axis=1)
        cv2.imwrite(os.path.join(save_dir, f'epoch{epoch:04d}_joint{j:02d}.jpg'), comparison)

    # Visualize PAF limbs
    for limb_k in range(paf_pred.shape[0] // 2):
        # GT PAF magnitude
        gt_px = paf_gt[limb_k * 2]
        gt_py = paf_gt[limb_k * 2 + 1]
        gt_mag = np.sqrt(gt_px ** 2 + gt_py ** 2).astype(np.float32)
        gt_norm = cv2.resize(gt_mag, (img_vis.shape[1], img_vis.shape[0]))
        gt_norm = cv2.normalize(gt_norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        gt_norm = cv2.applyColorMap(gt_norm, cv2.COLORMAP_HOT)
        gt_blend = cv2.addWeighted(img_vis, 0.5, gt_norm, 0.5, 0)

        # Pred PAF magnitude
        pred_px = paf_pred[limb_k * 2]
        pred_py = paf_pred[limb_k * 2 + 1]
        pred_mag = np.sqrt(pred_px ** 2 + pred_py ** 2).astype(np.float32)
        pred_norm = cv2.resize(pred_mag, (img_vis.shape[1], img_vis.shape[0]))
        pred_norm = cv2.normalize(pred_norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        pred_norm = cv2.applyColorMap(pred_norm, cv2.COLORMAP_HOT)
        pred_blend = cv2.addWeighted(img_vis, 0.5, pred_norm, 0.5, 0)

        comparison = np.concatenate([gt_blend, pred_blend], axis=1)
        cv2.imwrite(os.path.join(save_dir, f'epoch{epoch:04d}_paf_limb{limb_k:02d}.jpg'), comparison)


# --- Training Loop ------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build HRNet model with dual PAF+heatmap output
    model = HRNet(num_joints=17, num_limbs=16, width=args.width).to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"HRNet-W{args.width}: {num_params:.2f}M parameters")
    print(f"  Output: PAF 32-ch + Heatmap 17-ch")

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        if 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state)
        print(f"Resumed from {args.resume}")

    # Dataset and DataLoader
    train_ds = HRNetCocoDataset(
        args.data_dir, split="train2017",
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        sigma=args.sigma,
        paf_sigma=args.paf_sigma,
    )
    val_ds = HRNetCocoDataset(
        args.data_dir, split="val2017",
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        sigma=args.sigma,
        paf_sigma=args.paf_sigma,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True,
        prefetch_factor=4, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=4, persistent_workers=True,
    )

    # Loss, optimizer, scheduler
    criterion = HRNetLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # Cosine annealing with warmup (standard for HRNet)
    warmup_epochs = args.warmup_epochs
    if warmup_epochs > 0:
        def warmup_cosine(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
        )

    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda', enabled=True, init_scale=2 ** 16)

    # TensorBoard
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

        # ── Train ──
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch {epoch} start")
        last_step_finished_time = time.time()

        for step, (imgs, paf_gt, hm_gt, paf_mask) in enumerate(train_loader):
            imgs = imgs.to(device)
            paf_gt = paf_gt.to(device)
            hm_gt = hm_gt.to(device)
            paf_mask = paf_mask.to(device)

            with torch.amp.autocast('cuda'):
                paf_pred, hm_pred = model(imgs)
                # paf_pred = torch.clamp(paf_pred, -1.0, 1.0)
                # hm_pred = torch.clamp(hm_pred, 0, 1.0)
                loss = criterion(paf_pred, hm_pred, paf_gt, hm_gt, paf_mask)
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * args.accumulation_steps

            if (step + 1) % max(1, len(train_loader) // 5) == 0:
                now = time.time()
                step_time = now - last_step_finished_time
                steps_per_sec = max(1, len(train_loader) // 5) / step_time

                hm_max = hm_pred.max().item()
                hm_min = hm_pred.min().item()
                hm_mag_mean = torch.sqrt(hm_pred ** 2).mean().item()
                # 添加 PAF 数值稳定性检查
                paf_max = paf_pred.max().item()
                paf_min = paf_pred.min().item()
                paf_mag_mean = torch.sqrt(paf_pred ** 2).mean().item()
                print(
                    f"  Step {step + 1}/{len(train_loader)}  "
                    f"loss={loss.item() * args.accumulation_steps:.4f}  "
                    f"HM_range=[{hm_min:.4f}, {hm_max:.4f}] "
                    f"HM_mag_mean={hm_mag_mean:.4f} "
                    f"PAF_range=[{paf_min:.4f}, {paf_max:.4f}] "
                    f"PAF_mag_mean={paf_mag_mean:.4f} "
                    f"Mask_ratio={paf_mask.mean().item():.4f} "
                    f"steps/s={steps_per_sec:.1f}"
                )
                last_step_finished_time = now

        avg_train = train_loss / len(train_loader)

        if writer is not None:
            try:
                writer.add_scalar("loss/train", avg_train, epoch)
                writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
            except Exception:
                pass

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, paf_gt, hm_gt, paf_mask in val_loader:
                imgs = imgs.to(device)
                paf_gt = paf_gt.to(device)
                hm_gt = hm_gt.to(device)
                paf_mask = paf_mask.to(device)

                with torch.amp.autocast('cuda'):
                    paf_pred, hm_pred = model(imgs)
                    # paf_pred = torch.clamp(paf_pred, -1.0, 1.0)
                    # hm_pred = torch.clamp(hm_pred, 0, 1.0)
                    val_loss += criterion(paf_pred, hm_pred, paf_gt, hm_gt, paf_mask).item()

        avg_val = val_loss / len(val_loader)
        if writer is not None:
            try:
                writer.add_scalar("loss/val", avg_val, epoch)
            except Exception:
                pass

        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"train={avg_train:.4f}  val={avg_val:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"time={epoch_time:.1f}s"
        )

        # if avg_val < best_val:
        #     best_val = avg_val
        #     torch.save(
        #         model.state_dict(),
        #         os.path.join(
        #             args.save_dir,
        #             f"best_hrnet_w{args.width}_epoch{epoch:04d}_loss{best_val:.4f}.pth",
        #         ),
        #     )

        if epoch % args.save_every == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.save_dir,
                    f"hrnet_w{args.width}_epoch{epoch:04d}_loss{avg_val:.4f}.pth",
                ),
            )

        # Debug visualization
        if args.debug and epoch % args.debug_every == 0:
            model.eval()
            with torch.no_grad():
                imgs_vis, paf_gt_vis, hm_gt_vis, _ = next(iter(val_loader))
                imgs_vis = imgs_vis[:1].to(device)
                with torch.amp.autocast('cuda'):
                    paf_pred_vis, hm_pred_vis = model(imgs_vis)
                save_heatmap_comparison(
                    imgs_vis[0],
                    hm_pred_vis[0].cpu().numpy(),
                    hm_gt_vis[0].numpy(),
                    paf_pred_vis[0].cpu().numpy(),
                    paf_gt_vis[0].numpy(),
                    epoch,
                    DEBUG_OUTPUT_DIR,
                )

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
    parser = argparse.ArgumentParser(description="Train HRNet body pose model on COCO keypoints")

    # Data
    parser.add_argument("--data_dir", default="./data",
                        help="COCO dataset root directory")
    parser.add_argument("--input_size", type=int, default=256,
                        help="Model input image size (default: 256)")
    parser.add_argument("--heatmap_size", type=int, default=64,
                        help="Heatmap output size (default: 64, stride=4)")
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="Gaussian sigma for heatmap generation (default: 3.0)")
    parser.add_argument("--paf_sigma", type=float, default=2.0,
                        help="PAF limb width in heatmap-space pixels (default: 2.0)")

    # Model
    parser.add_argument("--width", type=int, default=32,
                        help="HRNet width: 32 for W32, 48 for W48 (default: 32)")

    # Training
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 210)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (default: 32)")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * accumulation_steps)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay (default: 1e-4)")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Warmup epochs for LR schedule (default: 5, 0 to disable)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader num_workers")

    # Checkpointing / Logging
    parser.add_argument("--save_dir", default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", default="runs/hrnet",
                        help="TensorBoard log directory")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    # parser.add_argument("--resume", default="./checkpoints/hrnet_w48_epoch0019_loss0.1242.pth",
    #                     help="Path to checkpoint to resume training from")
    parser.add_argument("--resume", default=None)

    # Debug
    parser.add_argument("--debug", action="store_true", default=True,
                        help="Enable debug heatmap visualization")
    parser.add_argument("--debug_every", type=int, default=1,
                        help="Save debug visualization every N epochs")

    args = parser.parse_args()
    train(args)
