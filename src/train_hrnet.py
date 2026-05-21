"""Training script for HRNet body pose model on COCO keypoints.

HRNet's dual-branch (PAF + heatmap) architecture for multi-person pose estimation.

Key differences from OpenPose training:
  - HRNet outputs 36-ch PAF + 18-ch heatmap (18 limbs, 18 keypoints)
  - Uses MSE loss on both PAF (with mask weighting) and heatmap
  - Uses Adam optimizer with warmup + cosine annealing (standard for HRNet)
  - ImageNet normalization instead of [0.0, 1.0]
  - Input size 256x256 (vs OpenPose's 368x368)

Dataset expected format (COCO-style):
  - Images in <data_dir>/images/<split>/
  - Annotations in <data_dir>/annotations/person_keypoints_<split>.json

Usage:
  python train_hrnet.py --data_dir /path/to/coco --epochs 150 --batch_size 32
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss.hrnet_loss import HRNetLoss
from models.hrnet_model import HRNet
from config import NUM_JOINTS, NUM_LIMBS, NUM_PAF_CHANNELS
from utils.HRNetCocoDataset import HRNetCocoDataset
from utils.visualization_util import save_heatmap_comparison

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def get_device():
    """Get device to use for training."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def execute_train(model, optimizer, epoch, train_loader, criterion, train_args):
    """Train for one epoch."""
    device = get_device()
    model.train()
    train_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", dynamic_ncols=True)
    for step, (imgs, paf_gt, hm_gt, paf_mask) in enumerate(pbar):
        imgs = imgs.to(device)
        paf_gt = paf_gt.to(device)
        hm_gt = hm_gt.to(device)
        paf_mask = paf_mask.to(device)

        paf_pred, hm_pred = model(imgs)
        loss = criterion(paf_pred, hm_pred, paf_gt, hm_gt, paf_mask)
        loss = loss / train_args.accumulation_steps

        # 检查 loss 是否为 NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  ⚠️  Warning: NaN/Inf loss detected at step {step}, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()

        if (step + 1) % train_args.accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss += loss.item() * train_args.accumulation_steps

        if (step + 1) % max(1, len(train_loader) // train_args.debug_num_every_epoch) == 0:
            if train_args.debug:
                save_heatmap_comparison(
                    imgs[0],
                    hm_pred[0].detach().cpu().numpy(),
                    hm_gt[0].cpu().numpy(),
                    paf_pred[0].detach().cpu().numpy(),
                    paf_gt[0].cpu().numpy(),
                    epoch=epoch,
                    step=step + 1,
                    save_dir=train_args.debug_output_dir
                )

            hm_gt_mag_mean = torch.sqrt(hm_gt ** 2).mean().item()
            hm_pred_mag_mean = torch.sqrt(hm_pred ** 2).mean().item()
            paf_gt_mag_mean = torch.sqrt(paf_gt ** 2).mean().item()
            paf_pred_mag_mean = torch.sqrt(paf_pred ** 2).mean().item()

            # 计算 Heatmap 相似度
            hm_cosine_sim = torch.nn.functional.cosine_similarity(
                hm_pred.flatten(start_dim=1),
                hm_gt.flatten(start_dim=1)
            ).mean().item()

            # 计算 PAF 相似度
            paf_cosine_sim = torch.nn.functional.cosine_similarity(
                paf_pred.flatten(start_dim=1),
                paf_gt.flatten(start_dim=1)
            ).mean().item()

            # 计算 MSE
            hm_mse = torch.mean((hm_pred - hm_gt) ** 2).item()
            paf_mse = torch.mean((paf_pred - paf_gt) ** 2).item()
            print(
                f"  loss={loss.item() * train_args.accumulation_steps:.5f}  "
                f"HM_mean_gt/pred={hm_gt_mag_mean:.4f}/{hm_pred_mag_mean:.4f} "
                f"HM_cos_sim={hm_cosine_sim:.4f} "
                f"HM_MSE={hm_mse:.6f}  "
                f"PAF_mean_gt/pred={paf_gt_mag_mean:.4f}/{paf_pred_mag_mean:.4f} "
                f"PAF_cos_sim={paf_cosine_sim:.4f} "
                f"PAF_MSE={paf_mse:.6f}"
            )
    pbar.close()
    return train_loss / len(train_loader)


def execute_validate(model, val_loader, criterion):
    """Run validation for one epoch."""
    device = get_device()
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, paf_gt, hm_gt, paf_mask in val_loader:
            imgs = imgs.to(device)
            paf_gt = paf_gt.to(device)
            hm_gt = hm_gt.to(device)
            paf_mask = paf_mask.to(device)

            paf_pred, hm_pred = model(imgs)
            val_loss += criterion(paf_pred, hm_pred, paf_gt, hm_gt, paf_mask).item()

    return val_loss / len(val_loader)


def do_train(args):
    """Training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build HRNet model with dual PAF+heatmap output
    model = HRNet(num_joints=NUM_JOINTS, num_limbs=NUM_LIMBS, width=args.width).to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"HRNet-W{args.width}: {num_params:.2f}M parameters")
    print(f"  Output: PAF {NUM_PAF_CHANNELS}-ch + Heatmap {NUM_JOINTS}-ch")

    # Loss, optimizer, scheduler
    criterion = HRNetLoss().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 👇 修改：将模型参数和 Loss 函数参数合并
    optimizer = optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Cosine annealing with warmup (standard for HRNet)
    warmup_epochs = args.warmup_epochs
    if warmup_epochs > 0:
        def warmup_cosine(current_epoch):
            if current_epoch < warmup_epochs:
                return (current_epoch + 1) / warmup_epochs
            progress = (current_epoch - warmup_epochs) / (args.epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ReduceLROnPlateau: 在 warmup 完成后接管学习率调度
    min_lr = args.lr * 0.01
    reduce_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
        min_lr=min_lr
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

        # 支持多种 checkpoint 格式
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Resumed from {args.resume}")

            start_epoch = 1
            best_val = float("inf")

            if not args.resume_and_reset:
                # 恢复 criterion 状态
                if 'criterion_state_dict' in checkpoint:
                    criterion.load_state_dict(checkpoint['criterion_state_dict'])
                    print("  → Criterion (Loss weights) state restored")
                # 恢复 optimizer 状态
                if 'optimizer' in checkpoint and hasattr(optimizer, 'load_state_dict'):
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("  → Optimizer state restored")
                # 恢复 scheduler 状态
                if 'scheduler' in checkpoint and hasattr(scheduler, 'load_state_dict'):
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    print("  → Scheduler state restored")
                # 恢复 ReduceLROnPlateau 状态
                if 'reduce_lr_scheduler' in checkpoint:
                    reduce_lr_scheduler.load_state_dict(checkpoint['reduce_lr_scheduler'])
                    print("  → ReduceLROnPlateau state restored")
                # 恢复 best_val 和 epoch
                start_epoch = checkpoint.get('epoch', 0) + 1
                if 'best_val' in checkpoint:
                    best_val = checkpoint['best_val']
                    print(f"  → Best val loss: {best_val:.5f}")
                else:
                    best_val = float("inf")
        else:
            # 旧的格式，直接是 state_dict
            model.load_state_dict(checkpoint)
            print(f"Resumed from {args.resume} (old format)")
            start_epoch = 1
            best_val = float("inf")

            # 恢复后重新设置学习率（warm restart）
            warm_restart_lr = args.lr * 0.3
            for param_group in optimizer.param_groups:
                param_group['lr'] = warm_restart_lr
            print(f"  → Learning rate set to {warm_restart_lr:.2e} (warm restart)")
    else:
        start_epoch = 1
        best_val = float("inf")

    # Dataset and DataLoader
    train_ds = HRNetCocoDataset(
        args.data_dir, split="train2017",
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        sigma=args.sigma,
        paf_sigma=args.paf_sigma,
        augment=args.augment,
        filter_key_points_nums=args.filter_key_points_nums
    )
    val_ds = HRNetCocoDataset(
        args.data_dir, split="val2017",
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        sigma=args.sigma,
        paf_sigma=args.paf_sigma,
        augment=False
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

    # TensorBoard
    writer = None
    if SummaryWriter is not None:
        try:
            writer = SummaryWriter(log_dir=args.log_dir)
        except Exception as e:
            print(f"Warning: Failed to initialize TensorBoard: {e}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Early stopping 配置
    patience = 10
    patience_counter = 0
    best_epoch = start_epoch - 1

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch {epoch}/{args.epochs} start")

        # ── Train ──
        avg_train = execute_train(model, optimizer, epoch, train_loader, criterion, args)

        # ── Validate ──
        avg_val = execute_validate(model, val_loader, criterion)

        if writer is not None:
            try:
                writer.add_scalar("loss/train", avg_train, epoch)
                writer.add_scalar("loss/val", avg_val, epoch)
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)

                # 记录不确定性加权参数
                # 记录 log_sigma 原始值
                writer.add_scalar("uncertainty/log_sigma_paf", criterion.log_sigma_paf.item(), epoch)
                writer.add_scalar("uncertainty/log_sigma_hm", criterion.log_sigma_hm.item(), epoch)

                # 记录实际生效的 Loss 权重 exp(-log_sigma)
                # 这个值越大约表示模型认为该任务越难，给予的权重越高
                writer.add_scalar("uncertainty/weight_paf", torch.exp(-criterion.log_sigma_paf).item(), epoch)
                writer.add_scalar("uncertainty/weight_hm", torch.exp(-criterion.log_sigma_hm).item(), epoch)
            except Exception:
                pass

        # 学习率调度：LambdaLR 始终调用（warmup + cosine annealing），
        # warmup 结束后额外调用 ReduceLROnPlateau 做自适应衰减
        scheduler.step()
        if epoch > warmup_epochs:
            reduce_lr_scheduler.step(avg_val)

        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch {epoch}/{args.epochs} end  "
            f"train={avg_train:.5f}  val={avg_val:.5f}  "
            f"lr={current_lr:.2e}  "
            f"time={epoch_time:.1f}s"
        )

        # 改进的 Best Model 保存策略
        if avg_val < best_val - 1e-5:
            best_val = avg_val
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict(),  # 新增：保存 Loss 参数
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'reduce_lr_scheduler': reduce_lr_scheduler.state_dict(),
                'best_val': best_val,
                'args': args,
            }, os.path.join(args.save_dir, "best.pth"))
            print(f"  → New best model saved (val={best_val:.5f})")
        else:
            patience_counter += 1
            print(f"  → No improvement for {patience_counter} epochs (best: {best_val:.5f} at epoch {best_epoch})")

            if patience_counter >= patience:
                print(f"\n{'=' * 60}")
                print(f"  Early stopping triggered at epoch {epoch}")
                print(f"  Best epoch: {best_epoch}, Best val loss: {best_val:.5f}")
                print(f"{'=' * 60}\n")
                break

        if epoch % args.save_every == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, f"hrnet_w{args.width}_epoch{epoch:04d}_loss{avg_val:.5f}.pth")
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if writer is not None:
        try:
            writer.close()
        except Exception:
            pass
    print(f"\n{'=' * 60}")
    print(f"Training complete.")
    print(f"Best epoch: {best_epoch}, Best val loss: {best_val:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HRNet multiple body pose model on COCO keypoints")

    # Data
    parser.add_argument("--data_dir", default="../data",
                        help="COCO dataset root directory")
    parser.add_argument("--augment", default=False,
                        help="Data augmentation")
    parser.add_argument("--input_size", type=int, default=256,
                        help="Model input image size (default: 256)")
    parser.add_argument("--heatmap_size", type=int, default=64,
                        help="Heatmap output size (default: 64, stride=4)")
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="Gaussian sigma for heatmap generation (default: 2.0)")
    parser.add_argument("--paf_sigma", type=float, default=3.0,
                        help="PAF limb width in heatmap-space pixels (default: 3.0)")
    parser.add_argument("--filter_key_points_nums", type=int, default=10,
                        help="Drop key point num less then")

    # Model
    parser.add_argument("--width", type=int, default=32,
                        help="HRNet width: 32 for W32, 48 for W48 (default: 32)")

    # Training
    parser.add_argument("--epochs", type=int, default=150,
                        help="Number of training epochs (default: 150)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--accumulation_steps", type=int, default=2,
                        help="Gradient accumulation steps (effective batch = batch_size * accumulation_steps)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay (default: 1e-4)")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Warmup epochs for LR schedule (default: 10, 0 to disable)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader num_workers")

    # Checkpointing / Logging
    parser.add_argument("--save_dir", default="../checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", default="../runs/hrnet",
                        help="TensorBoard log directory")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save checkpoint every N epochs (default: 1)")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--resume_and_reset", default=False,
                        help="Resume from checkpoint and reset train")

    # Debug
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug visualization (default: False)")
    parser.add_argument("--debug_num_every_epoch", type=int, default=5,
                        help="Save debug visualization times every epoch (default: 5)")
    parser.add_argument("--debug_output_dir", default="../output/hrnet_train",
                        help="Save debug visualization directory")

    args = parser.parse_args()
    # 以下列参数覆盖默认参数
    args.augment = True
    args.sigma = 1.0
    args.paf_sigma = 1.0
    args.epochs = 120
    args.batch_size = 32
    args.accumulation_steps = 16
    args.lr = 5e-4
    args.weight_decay = 1e-4
    args.warmup_epochs = 15
    args.width = 48
    args.resume = "../checkpoints/best.pth"
    args.resume_and_reset = False
    args.debug = True
    args.debug_num_every_epoch = 5
    args.filter_key_points_nums = 10
    # args.workers = 12
    # args.log_dir = "/root/tf-logs/hrnet"
    # args.data_dir = "/root/autodl-tmp/data"

    do_train(args)
