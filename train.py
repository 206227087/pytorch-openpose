"""
Training script for pytorch-openpose body pose model.

Dataset expected format (COCO-style):
  - Images in <data_dir>/images/
  - Annotations in <data_dir>/annotations/person_keypoints_train2017.json

Usage:
  python train.py --data_dir /path/to/coco --epochs 100 --batch_size 8
"""

import argparse
import json
import os
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from src.model import bodypose_model


# ─── Multi-stage wrapper ───────────────────────────────────────────────────────

class BodyPoseTrainModel(bodypose_model):
    """Returns per-stage (PAF, heatmap) pairs for intermediate supervision."""

    def forward(self, x):
        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        paf_stages = [out1_1, out2_1, out3_1, out4_1, out5_1, out6_1]
        hm_stages = [out1_2, out2_2, out3_2, out4_2, out5_2, out6_2]
        return paf_stages, hm_stages


# ─── Dataset ──────────────────────────────────────────────────────────────────

# COCO keypoint order → OpenPose order mapping
# COCO: nose(0) l_eye(1) r_eye(2) l_ear(3) r_ear(4)
#       l_shoulder(5) r_shoulder(6) l_elbow(7) r_elbow(8)
#       l_wrist(9) r_wrist(10) l_hip(11) r_hip(12)
#       l_knee(13) r_knee(14) l_ankle(15) r_ankle(16)
# OpenPose 18 joints:
#   0:nose, 1:neck, 2:r_shoulder, 3:r_elbow, 4:r_wrist,
#   5:l_shoulder, 6:l_elbow, 7:l_wrist, 8:r_hip, 9:r_knee,
#   10:r_ankle, 11:l_hip, 12:l_knee, 13:l_ankle,
#   14:r_eye, 15:l_eye, 16:r_ear, 17:l_ear
COCO_TO_OPENPOSE = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

# PAF limb connections (pairs of OpenPose joint indices)
# Based on official OpenPose: https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 19 limbs total for COCO 18 joints (including neck)
LIMBS = [
    (1, 2), (1, 5),  # neck to shoulders (2)
    (2, 3), (3, 4),  # right arm (2)
    (5, 6), (6, 7),  # left arm (2)
    (1, 8), (1, 11),  # neck to hips (2)
    (8, 9), (9, 10),  # right leg (2)
    (11, 12), (12, 13),  # left leg (2)
    (1, 0),  # neck to nose (1)
    (0, 14), (14, 16),  # right eye-ear (2)
    (0, 15), (15, 17),  # left eye-ear (2)
    (2, 16), (5, 17),  # shoulders to ears (2) - MISSING!
]  # 19 limbs → 38 PAF channels (x,y per limb)

INPUT_SIZE = 368
HEATMAP_SIZE = INPUT_SIZE // 8  # 46
SIGMA = 7.0  # Gaussian spread for heatmaps
PAF_SIGMA = 8.0  # PAF limb width in pixels (heatmap space)
NUM_LIMBS = len(LIMBS)  # 19
NUM_PAF_CHANNELS = NUM_LIMBS * 2  # 38
NUM_JOINTS = 18


def make_heatmap(joints, size, sigma):
    """Generate Gaussian heatmaps for all 18 joints + background using vectorized operations."""
    hm = np.zeros((NUM_JOINTS + 1, size, size), dtype=np.float32)
    scale = size / INPUT_SIZE

    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:size, 0:size]

    for j_idx, (x, y, v) in enumerate(joints):
        if v == 0 or j_idx >= NUM_JOINTS:
            continue
        cx, cy = x * scale, y * scale

        # Vectorized Gaussian computation
        d2 = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        gaussian = np.exp(-d2 / (2 * sigma ** 2))

        # Take max if multiple joints overlap
        hm[j_idx] = np.maximum(hm[j_idx], gaussian)

    # background channel
    hm[NUM_JOINTS] = np.clip(1.0 - hm[:NUM_JOINTS].max(axis=0), 0, 1)
    return hm


def make_paf(joints, size, sigma):
    """Generate Part Affinity Fields for 19 limbs (38 channels) using vectorized operations."""
    paf = np.zeros((NUM_PAF_CHANNELS, size, size), dtype=np.float32)
    mask = np.zeros((NUM_PAF_CHANNELS, size, size), dtype=np.float32)
    scale = size / INPUT_SIZE

    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:size, 0:size]

    for limb_idx, (ja, jb) in enumerate(LIMBS):
        # Safety check: ensure joint indices are within bounds
        if ja >= len(joints) or jb >= len(joints):
            continue

        xa, ya, va = joints[ja]
        xb, yb, vb = joints[jb]
        if va == 0 or vb == 0:
            continue

        xa, ya = xa * scale, ya * scale
        xb, yb = xb * scale, yb * scale

        dx, dy = xb - xa, yb - ya
        length = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
        ux, uy = dx / length, dy / length  # unit vector

        # Vectorized distance computation
        # Perpendicular distance from point to line segment
        vec_x = x_grid - xa
        vec_y = y_grid - ya

        # Parallel distance (projection onto limb direction)
        d_par = vec_x * ux + vec_y * uy

        # Perpendicular distance
        d_perp = np.abs(vec_x * (-uy) + vec_y * ux)

        # Create mask for valid region
        valid_mask = (d_perp <= sigma) & (d_par >= 0) & (d_par <= length)

        # Assign PAF vectors where valid
        paf[limb_idx * 2, valid_mask] = ux
        paf[limb_idx * 2 + 1, valid_mask] = uy
        mask[limb_idx * 2, valid_mask] = 1
        mask[limb_idx * 2 + 1, valid_mask] = 1

    return paf, mask


class CocoKeypoints(Dataset):
    def __init__(self, data_dir, split="train2017", preprocessed_dir=None):
        self.preprocessed_dir = preprocessed_dir

        if preprocessed_dir and os.path.exists(preprocessed_dir):
            # Load preprocessed data
            self.split_dir = os.path.join(preprocessed_dir, split)
            self.npz_files = [f for f in os.listdir(self.split_dir) if f.endswith('.npz')]
            self.image_ids = [int(f.replace('.npz', '')) for f in self.npz_files]
            self.img_dir = os.path.join(data_dir, "images", split)

            # Load image file names from original annotation
            ann_file = os.path.join(data_dir, "annotations",
                                    f"person_keypoints_{split}.json")
            with open(ann_file) as f:
                data = json.load(f)
            self.id2file = {img["id"]: img["file_name"] for img in data["images"]}

            print(f"Loaded {len(self.image_ids)} preprocessed samples from {self.split_dir}")
        else:
            # Original loading logic
            ann_file = os.path.join(data_dir, "annotations",
                                    f"person_keypoints_{split}.json")
            with open(ann_file) as f:
                data = json.load(f)

            self.img_dir = os.path.join(data_dir, "images", split)
            self.id2file = {img["id"]: img["file_name"] for img in data["images"]}

            # group annotations by image, keep only images with ≥1 person
            self.samples = {}
            for ann in data["annotations"]:
                if ann.get("num_keypoints", 0) == 0:
                    continue
                iid = ann["image_id"]
                self.samples.setdefault(iid, []).append(ann)
            self.image_ids = list(self.samples.keys())

            print(f"Loaded {len(self.image_ids)} images for {split} (live preprocessing)")
            print(f"NUM_LIMBS={NUM_LIMBS}, NUM_PAF_CHANNELS={NUM_PAF_CHANNELS}, NUM_JOINTS={NUM_JOINTS}")

    def __len__(self):
        return len(self.image_ids)

    def _load_joints(self, ann):
        """Return list of (x, y, visibility) in OpenPose order."""
        kps = ann["keypoints"]  # flat [x,y,v, x,y,v, ...]
        coco_joints = [(kps[i * 3], kps[i * 3 + 1], kps[i * 3 + 2]) for i in range(17)]
        joints = []
        for op_idx in COCO_TO_OPENPOSE:
            if op_idx == -1:
                # Compute neck as midpoint of shoulders (COCO: l_shoulder=5, r_shoulder=6)
                l_shoulder = coco_joints[5]
                r_shoulder = coco_joints[6]
                if l_shoulder[2] > 0 and r_shoulder[2] > 0:
                    neck_x = (l_shoulder[0] + r_shoulder[0]) / 2
                    neck_y = (l_shoulder[1] + r_shoulder[1]) / 2
                    neck_v = 2  # visible
                else:
                    neck_x, neck_y, neck_v = 0, 0, 0
                joints.append((neck_x, neck_y, neck_v))
            else:
                joints.append(coco_joints[op_idx])
        return joints  # 18 joints

    def __getitem__(self, idx):
        if self.preprocessed_dir:
            # Load preprocessed data
            iid = self.image_ids[idx]
            npz_path = os.path.join(self.split_dir, f"{iid}.npz")
            data = np.load(npz_path)

            paf_t = torch.from_numpy(data['paf'])
            hm_t = torch.from_numpy(data['hm'])
            mask_t = torch.from_numpy(data['mask'])

            # Load and preprocess image
            img_path = os.path.join(self.img_dir, self.id2file[iid])
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))

            img = img.astype(np.float32) / 255.0 - 0.5
            img = torch.from_numpy(img.transpose(2, 0, 1))

            return img, paf_t, hm_t, mask_t
        else:
            # Original live preprocessing logic
            iid = self.image_ids[idx]
            img_path = os.path.join(self.img_dir, self.id2file[iid])
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)

            h, w = img.shape[:2]
            scale_x, scale_y = INPUT_SIZE / w, INPUT_SIZE / h
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))

            # aggregate heatmaps / PAFs across all persons in image
            hm_agg = np.zeros((NUM_JOINTS + 1, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
            paf_agg = np.zeros((NUM_PAF_CHANNELS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
            mask_agg = np.zeros((NUM_PAF_CHANNELS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)

            for ann in self.samples[iid]:
                joints = self._load_joints(ann)

                # Safety check: ensure we have exactly 18 joints
                if len(joints) != NUM_JOINTS:
                    print(f"Warning: Expected {NUM_JOINTS} joints, got {len(joints)}. Skipping this annotation.")
                    continue

                # scale joints to INPUT_SIZE space
                scaled = [(x * scale_x, y * scale_y, v) for x, y, v in joints]

                hm = make_heatmap(scaled, HEATMAP_SIZE, SIGMA)
                paf, pmask = make_paf(scaled, HEATMAP_SIZE, PAF_SIGMA)

                # Verify shapes before aggregation
                if hm.shape != hm_agg.shape:
                    print(f"Warning: Heatmap shape mismatch. Expected {hm_agg.shape}, got {hm.shape}")
                    continue
                if paf.shape != paf_agg.shape:
                    print(f"Warning: PAF shape mismatch. Expected {paf_agg.shape}, got {paf.shape}")
                    print(f"  NUM_PAF_CHANNELS={NUM_PAF_CHANNELS}, paf.ndim={paf.ndim}")
                    continue

                np.maximum(hm_agg, hm, out=hm_agg)

                # Fix: Preserve PAF direction by comparing magnitudes per limb
                # paf shape: (38, H, W) -> 19 limbs × 2 channels (x, y)
                # Compare limb magnitudes and update both x,y channels together
                for limb_idx in range(NUM_LIMBS):
                    # Calculate magnitude for this limb
                    paf_limb_mag = np.sqrt(paf[limb_idx * 2] ** 2 + paf[limb_idx * 2 + 1] ** 2)
                    agg_limb_mag = np.sqrt(paf_agg[limb_idx * 2] ** 2 + paf_agg[limb_idx * 2 + 1] ** 2)

                    # Update where new PAF has stronger magnitude
                    update_mask = paf_limb_mag > agg_limb_mag
                    paf_agg[limb_idx * 2][update_mask] = paf[limb_idx * 2][update_mask]
                    paf_agg[limb_idx * 2 + 1][update_mask] = paf[limb_idx * 2 + 1][update_mask]

                np.maximum(mask_agg, pmask, out=mask_agg)

            # normalize image
            img = img.astype(np.float32) / 255.0 - 0.5
            img = torch.from_numpy(img.transpose(2, 0, 1))  # (3, H, W)

            hm_t = torch.from_numpy(hm_agg)
            paf_t = torch.from_numpy(paf_agg)
            mask_t = torch.from_numpy(mask_agg)

            return img, paf_t, hm_t, mask_t


# ─── Loss ─────────────────────────────────────────────────────────────────────

class OpenPoseLoss(nn.Module):
    """MSE loss summed over all 6 stages, weighted by PAF mask."""

    def forward(self, paf_stages, hm_stages, paf_gt, hm_gt, paf_mask):
        loss = 0.0
        for paf_pred, hm_pred in zip(paf_stages, hm_stages):
            # PAF loss: only penalise where a limb is present
            paf_diff = (paf_pred - paf_gt) * paf_mask
            loss += (paf_diff ** 2).mean()
            loss += ((hm_pred - hm_gt) ** 2).mean()
        return loss


# ─── Training loop ────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # model
    model = BodyPoseTrainModel().to(device)

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state)
        print(f"Resumed from {args.resume}")

    # data
    train_ds = CocoKeypoints(args.data_dir, split="train2017", preprocessed_dir=args.preprocessed_dir)
    val_ds = CocoKeypoints(args.data_dir, split="val2017", preprocessed_dir=args.preprocessed_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=True)

    criterion = OpenPoseLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(args.epochs * 0.6),
                               int(args.epochs * 0.85)], gamma=0.1)

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    writer = None
    if SummaryWriter is not None:
        try:
            writer = SummaryWriter(log_dir=args.log_dir)
        except Exception as e:
            print(f"Warning: Failed to initialize TensorBoard: {e}")
            print("Continuing without TensorBoard logging...")

    os.makedirs(args.save_dir, exist_ok=True)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch {epoch} start")
        for step, (imgs, paf_gt, hm_gt, paf_mask) in enumerate(train_loader):
            imgs = imgs.to(device)
            paf_gt = paf_gt.to(device)
            hm_gt = hm_gt.to(device)
            paf_mask = paf_mask.to(device)
            # Forward pass with automatic mixed precision
            with torch.cuda.amp.autocast():
                paf_stages, hm_stages = model(imgs)
                loss = criterion(paf_stages, hm_stages, paf_gt, hm_gt, paf_mask)
            # Backward pass with scaling
            scaler.scale(loss).backward()
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            if step % 100 == 0:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  Epoch {epoch} step {step}/{len(train_loader)}"
                      f"  loss={loss.item():.4f}")

        avg_train = train_loss / len(train_loader)
        if writer is not None:
            try:
                writer.add_scalar("loss/train", avg_train, epoch)
            except:
                pass

        # ── validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, paf_gt, hm_gt, paf_mask in val_loader:
                imgs = imgs.to(device)
                paf_gt = paf_gt.to(device)
                hm_gt = hm_gt.to(device)
                paf_mask = paf_mask.to(device)

                # Use autocast in validation as well for consistency and memory saving
                with torch.cuda.amp.autocast():
                    paf_stages, hm_stages = model(imgs)
                    val_loss += criterion(paf_stages, hm_stages,
                                          paf_gt, hm_gt, paf_mask).item()

        avg_val = val_loss / len(val_loader)
        if writer is not None:
            try:
                writer.add_scalar("loss/val", avg_val, epoch)
            except:
                pass

        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs}  "
              f"train={avg_train:.4f}  val={avg_val:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # save best
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f"best_epoch{epoch:04d}_loss{best_val:.4f}.pth"))

        # periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f"epoch{epoch:04d}_loss{avg_val:.4f}.pth"))

    if writer is not None:
        try:
            writer.close()
        except:
            pass
    print("Training complete. Best val loss:", best_val)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--preprocessed_dir", default="./preprocessed")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--log_dir", default="runs")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--resume", default="./checkpoints/epoch0002_val0.3889.pth")
    args = parser.parse_args()
    train(args)
