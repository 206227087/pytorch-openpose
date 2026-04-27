"""
@Author: chaos
@Date: 2026/4/26
@Version：V1.0 
@Description：
"""
"""COCO key points dataset.

Supports two modes:
1. Pre-processed: loads .npz files from preprocessed_dir (fast)
2. Live: generates heatmaps/PAFs on-the-fly from COCO JSON annotations
"""

import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import (
    COCO_TO_OPENPOSE, LIMBS, NUM_LIMBS, NUM_PAF_CHANNELS, NUM_JOINTS,
    INPUT_SIZE, HEATMAP_SIZE, SIGMA, PAF_SIGMA
)
from src.preprocessing import normalize_image


def make_heatmap(joints, size, sigma):
    """Generate Gaussian heatmaps for all 18 joints + background.

    Args:
        joints: List of (x, y, visibility) tuples in OpenPose order
        size: Heatmap spatial size (e.g., 46)
        sigma: Gaussian spread in heatmap-space pixels

    Returns:
        hm: (19, size, size) float32 array (18 joints + background)
    """
    hm = np.zeros((NUM_JOINTS + 1, size, size), dtype=np.float32)
    scale = size / INPUT_SIZE
    y_grid, x_grid = np.mgrid[0:size, 0:size]

    for j_idx, (x, y, v) in enumerate(joints):
        if v == 0 or j_idx >= NUM_JOINTS:
            continue
        cx, cy = x * scale, y * scale
        g = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2))
        np.maximum(hm[j_idx], g, out=hm[j_idx])

    hm[NUM_JOINTS] = np.clip(1.0 - hm[:NUM_JOINTS].max(axis=0), 0, 1)
    return hm


def make_paf(joints, size, sigma):
    """Generate Part Affinity Fields for all 19 limbs (38 channels).

    Args:
        joints: List of (x, y, visibility) tuples in OpenPose order
        size: PAF spatial size (e.g., 46)
        sigma: PAF limb width in heatmap-space pixels

    Returns:
        paf: (38, size, size) float32 array
        mask: (38, size, size) float32 array (1 where limb is present)
    """
    paf = np.zeros((NUM_PAF_CHANNELS, size, size), dtype=np.float32)
    mask = np.zeros((NUM_PAF_CHANNELS, size, size), dtype=np.float32)
    scale = size / INPUT_SIZE

    y_grid, x_grid = np.mgrid[0:size, 0:size]

    for limb_idx, (ja, jb) in enumerate(LIMBS):
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
        ux, uy = dx / length, dy / length

        vec_x = x_grid - xa
        vec_y = y_grid - ya
        d_par = vec_x * ux + vec_y * uy
        d_perp = np.abs(vec_x * (-uy) + vec_y * ux)

        valid_mask = (d_perp <= sigma) & (d_par >= 0) & (d_par <= length)

        paf[limb_idx * 2, valid_mask] = ux
        paf[limb_idx * 2 + 1, valid_mask] = uy
        mask[limb_idx * 2, valid_mask] = 1
        mask[limb_idx * 2 + 1, valid_mask] = 1

    return paf, mask


def load_joints(ann, orig_w, orig_h):
    """Convert COCO keypoints to OpenPose 18-joint format.

    Computes neck as midpoint of left and right shoulders.
    Returns list of (x, y, visibility) tuples.
    """
    kps = ann["keypoints"]
    coco_joints = []

    sx = INPUT_SIZE / orig_w
    sy = INPUT_SIZE / orig_h

    for i in range(17):
        x, y, v = kps[i * 3], kps[i * 3 + 1], kps[i * 3 + 2]
        if v > 0:
            x = x * sx
            y = y * sy
            x = min(max(x, 0), INPUT_SIZE - 1)
            y = min(max(y, 0), INPUT_SIZE - 1)
        coco_joints.append((x, y, v))

    l_shoulder = coco_joints[5]
    r_shoulder = coco_joints[6]
    if l_shoulder[2] > 0 and r_shoulder[2] > 0:
        neck = (
            (l_shoulder[0] + r_shoulder[0]) / 2,
            (l_shoulder[1] + r_shoulder[1]) / 2,
            min(l_shoulder[2], r_shoulder[2]),
        )
    else:
        neck = (0, 0, 0)

    joints = []
    for idx in COCO_TO_OPENPOSE:
        if idx == -1:
            joints.append(neck)
        else:
            joints.append(coco_joints[idx])

    return joints


class CustomDataSet(Dataset):
    """COCO key points dataset

    Supports two modes:
    1. Pre-processed: loads .npz files from preprocessed_dir (fast)
    2. Live: generates heatmaps/PAFs on-the-fly from COCO JSON annotations
    """

    def __init__(self, data_dir, split="train2017", preprocessed_dir=None, load_to_memory=False):
        self.preprocessed_dir = preprocessed_dir
        self.load_to_memory = load_to_memory
        # Load from npz
        self.npz_paths = {}
        self.npz_data = {}
        # Load from COCO JSON
        self.id2file = {}
        self.img_dir = os.path.join(data_dir, "images", split)

        if preprocessed_dir and os.path.exists(preprocessed_dir):
            self.split_dir = os.path.join(preprocessed_dir, split)
            self.npz_files = [f for f in os.listdir(self.split_dir) if f.endswith('.npz')]
            self.image_ids = [int(f.replace('.npz', '')) for f in self.npz_files]

            self.npz_paths = {
                iid: os.path.join(self.split_dir, f"{iid}.npz")
                for iid in self.image_ids
            }

            if load_to_memory:
                print(f"Loading all {len(self.image_ids)} samples into memory...")
                for iid in self.image_ids:
                    npz = np.load(self.npz_paths[iid])
                    self.npz_data[iid] = {
                        'img_path': npz["img_path"].tobytes().decode('utf-8'),
                        'paf': npz["paf"],
                        'hm': npz["hm"],
                        'mask': npz["mask"]
                    }
                print(f"NPZ data loaded!")
            print(f"Loaded {len(self.image_ids)} preprocessed samples from {self.split_dir}")
        else:
            ann_file = os.path.join(data_dir, "annotations",
                                    f"person_keypoints_{split}.json")
            with open(ann_file) as f:
                data = json.load(f)

            self.id2file = {img["id"]: img["file_name"] for img in data["images"]}

            self.samples = {}
            for ann in data["annotations"]:
                if ann.get("num_keypoints", 0) == 0:
                    continue
                iid = ann["image_id"]
                self.samples.setdefault(iid, []).append(ann)
            self.image_ids = list(self.samples.keys())
            print(f"Loaded {len(self.image_ids)} images with annotations")

    def __len__(self):
        return len(self.image_ids)

    def __del__(self):
        """Cleanup resources."""
        self.npz_data.clear()

    def load_image_from_npz(self, idx):
        iid = self.image_ids[idx]
        if self.load_to_memory:
            npz_data = self.npz_data[iid]
            img_path = npz_data['img_path']
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
            if img is None:
                img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = normalize_image(img)
            img = torch.from_numpy(img.transpose(2, 0, 1))

            paf_t = torch.from_numpy(npz_data['paf'].astype(np.float32))
            hm_t = torch.from_numpy(npz_data['hm'].astype(np.float32))
            mask_t = torch.from_numpy(npz_data['mask'].astype(np.float32))
            return img, paf_t, hm_t, mask_t
        else:
            npz_data = np.load(self.npz_paths[iid], mmap_mode='r')
            img_path = npz_data["img_path"].tobytes().decode('utf-8')
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
            if img is None:
                img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = normalize_image(img)
            img = torch.from_numpy(img.transpose(2, 0, 1))

            paf_t = torch.from_numpy(npz_data["paf"].astype(np.float32))
            hm_t = torch.from_numpy(npz_data["hm"].astype(np.float32))
            mask_t = torch.from_numpy(npz_data["mask"].astype(np.float32))

            return img, paf_t, hm_t, mask_t

    def load_image_from_coco(self, idx):
        iid = self.image_ids[idx]
        img_file = os.path.join(self.img_dir, self.id2file[iid])
        img = cv2.imread(img_file)
        orig_w, orig_h = img.shape[1], img.shape[0]
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        annotations = self.samples[iid]

        hm_agg = np.zeros((NUM_JOINTS + 1, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
        paf_agg = np.zeros((NUM_PAF_CHANNELS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
        mask_agg = np.zeros((NUM_PAF_CHANNELS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)

        for ann in annotations:
            joints = load_joints(ann, orig_w, orig_h)
            scaled = [(x, y, v) for x, y, v in joints]

            hm = make_heatmap(scaled, HEATMAP_SIZE, SIGMA)
            paf, pmask = make_paf(scaled, HEATMAP_SIZE, PAF_SIGMA)

            if hm.shape != hm_agg.shape:
                continue
            if paf.shape != paf_agg.shape:
                continue

            np.maximum(hm_agg, hm, out=hm_agg)

            for limb_idx in range(NUM_LIMBS):
                paf_limb_mag = np.sqrt(paf[limb_idx * 2] ** 2 + paf[limb_idx * 2 + 1] ** 2)
                agg_limb_mag = np.sqrt(paf_agg[limb_idx * 2] ** 2 + paf_agg[limb_idx * 2 + 1] ** 2)
                update_mask = paf_limb_mag > agg_limb_mag
                paf_agg[limb_idx * 2][update_mask] = paf[limb_idx * 2][update_mask]
                paf_agg[limb_idx * 2 + 1][update_mask] = paf[limb_idx * 2 + 1][update_mask]

            np.maximum(mask_agg, pmask, out=mask_agg)

        img = normalize_image(img)
        img = torch.from_numpy(img.transpose(2, 0, 1))

        hm_t = torch.from_numpy(hm_agg)
        paf_t = torch.from_numpy(paf_agg)
        mask_t = torch.from_numpy(mask_agg)

        return img, paf_t, hm_t, mask_t

    def __getitem__(self, idx):
        if self.preprocessed_dir and os.path.exists(self.preprocessed_dir):
            return self.load_image_from_npz(idx)
        else:
            return self.load_image_from_coco(idx)
