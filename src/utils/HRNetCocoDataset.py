"""COCO keypoint dataset for HRNet training.

Generates COCO-style heatmap targets (17 keypoints) AND Part Affinity Field
(PAF) targets (16 limbs x 2 = 32 channels) with masks, matching the
dual-branch HRNet output format for multi-person pose estimation.

Dataset expected format (COCO-style):
  - Images in <data_dir>/images/<split>/
  - Annotations in <data_dir>/annotations/person_keypoints_<split>.json
"""

import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# COCO 17 keypoints:
# 0:nose 1:left_eye 2:right_eye 3:left_ear 4:right_ear
# 5:left_shoulder 6:right_shoulder 7:left_elbow 8:right_elbow
# 9:left_wrist 10:right_wrist 11:left_hip 12:right_hip
# 13:left_knee 14:right_knee 15:left_ankle 16:right_ankle
NUM_COCO_JOINTS = 17

# COCO skeleton connections (16 limbs)
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),         # face
    (5, 6), (5, 7), (7, 9), (6, 8),         # arms
    (8, 10), (5, 11), (6, 12),              # torso
    (11, 12), (11, 13), (13, 15),           # left leg
    (12, 14), (14, 16),                     # right leg
]
NUM_COCO_LIMBS = len(COCO_SKELETON)  # 16
NUM_PAF_CHANNELS = NUM_COCO_LIMBS * 2  # 32

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_hrnet_heatmap(joints, size, sigma):
    """Generate Gaussian heatmaps for 17 COCO keypoints.

    Args:
        joints: list of (x, y, visibility) tuples, length 17.
        size: heatmap spatial size (e.g., 64 for 256x256 input with stride 4).
        sigma: Gaussian spread in heatmap-space pixels.

    Returns:
        hm: (17, size, size) float32 array.
    """
    hm = np.zeros((NUM_COCO_JOINTS, size, size), dtype=np.float32)
    y_grid, x_grid = np.mgrid[0:size, 0:size]

    for j_idx, (x, y, v) in enumerate(joints):
        if v == 0 or j_idx >= NUM_COCO_JOINTS:
            continue
        g = np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma ** 2))
        np.maximum(hm[j_idx], g, out=hm[j_idx])

    return hm


def make_hrnet_paf(joints, size, sigma):
    """Generate Part Affinity Fields for 16 COCO limbs (32 channels).

    Args:
        joints: list of (x, y, visibility) tuples in heatmap-space coords.
        size: PAF spatial size (e.g., 64).
        sigma: PAF limb width in heatmap-space pixels.

    Returns:
        paf: (32, size, size) float32 array.
        mask: (32, size, size) float32 array (1 where limb is present).
    """
    paf = np.zeros((NUM_PAF_CHANNELS, size, size), dtype=np.float32)
    mask = np.zeros((NUM_PAF_CHANNELS, size, size), dtype=np.float32)
    y_grid, x_grid = np.mgrid[0:size, 0:size]

    for limb_idx, (ja, jb) in enumerate(COCO_SKELETON):
        if ja >= len(joints) or jb >= len(joints):
            continue

        xa, ya, va = joints[ja]
        xb, yb, vb = joints[jb]
        if va == 0 or vb == 0:
            continue

        dx, dy = xb - xa, yb - ya
        length = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
        ux, uy = dx / length, dy / length

        # Vector from point A to each grid point
        vec_x = x_grid - xa
        vec_y = y_grid - ya

        # Parallel and perpendicular distances
        d_par = vec_x * ux + vec_y * uy
        d_perp = np.abs(vec_x * (-uy) + vec_y * ux)

        # Valid region: within limb width and along limb length
        valid_mask = (d_perp <= sigma) & (d_par >= 0) & (d_par <= length)

        paf[limb_idx * 2, valid_mask] = ux
        paf[limb_idx * 2 + 1, valid_mask] = uy
        mask[limb_idx * 2, valid_mask] = 1.0
        mask[limb_idx * 2 + 1, valid_mask] = 1.0

    return paf, mask


def load_coco_joints(ann, input_size, orig_w, orig_h):
    """Load COCO 17 keypoints from annotation, scaled to input_size.

    Args:
        ann: COCO annotation dict with 'keypoints' field.
        input_size: target image size (e.g., 256).
        orig_w, orig_h: original image dimensions.

    Returns:
        List of (x, y, visibility) tuples in input-space coordinates.
    """
    kps = ann["keypoints"]
    sx = input_size / orig_w
    sy = input_size / orig_h

    joints = []
    for i in range(NUM_COCO_JOINTS):
        x, y, v = kps[i * 3], kps[i * 3 + 1], kps[i * 3 + 2]
        if v > 0:
            x = x * sx
            y = y * sy
            x = min(max(x, 0), input_size - 1)
            y = min(max(y, 0), input_size - 1)
        joints.append((x, y, v))

    return joints


class HRNetCocoDataset(Dataset):
    """COCO keypoint dataset for HRNet multi-person training.

    Generates 17-channel heatmap targets AND 32-channel PAF targets with
    masks from COCO annotations. Images are resized to input_size x
    input_size and normalized with ImageNet mean/std.

    Returns (img, paf_gt, hm_gt, paf_mask) matching the same format as
    CustomDataSet used by OpenPose training.

    Args:
        data_dir: COCO dataset root directory.
        split: data split (e.g., 'train2017', 'val2017').
        input_size: model input image size (default 256).
        heatmap_size: output heatmap size (default 64, i.e. stride=4).
        sigma: Gaussian spread for heatmap generation (default 2.0).
        paf_sigma: PAF limb width in heatmap-space pixels (default 2.0).
    """

    def __init__(self, data_dir, split="train2017",
                 input_size=256, heatmap_size=64, sigma=2.0, paf_sigma=2.0):
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.paf_sigma = paf_sigma
        self.scale = heatmap_size / input_size

        self.img_dir = os.path.join(data_dir, "images", split)
        ann_file = os.path.join(data_dir, "annotations",
                                f"person_keypoints_{split}.json")

        with open(ann_file) as f:
            data = json.load(f)

        self.id2file = {img["id"]: img["file_name"] for img in data["images"]}
        self.id2size = {img["id"]: (img["width"], img["height"]) for img in data["images"]}

        # Group annotations by image_id, filter out empty annotations
        self.samples = {}
        for ann in data["annotations"]:
            if ann.get("num_keypoints", 0) == 0:
                continue
            iid = ann["image_id"]
            if iid not in self.samples:
                self.samples[iid] = []
            self.samples[iid].append(ann)

        self.image_ids = sorted(self.samples.keys())
        print(f"Loaded {len(self.image_ids)} images with keypoints for {split}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        iid = self.image_ids[idx]
        img_file = os.path.join(self.img_dir, self.id2file[iid])
        orig_w, orig_h = self.id2size[iid]

        # Load and resize image
        img = cv2.imread(img_file)
        if img is None:
            img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.input_size, self.input_size),
                             interpolation=cv2.INTER_LINEAR)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Normalize with ImageNet mean/std
        img_float = img.astype(np.float32) / 255.0
        img_float = (img_float - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1))

        # Generate aggregated heatmap and PAF from all person annotations
        hm_agg = np.zeros((NUM_COCO_JOINTS, self.heatmap_size, self.heatmap_size),
                           dtype=np.float32)
        paf_agg = np.zeros((NUM_PAF_CHANNELS, self.heatmap_size, self.heatmap_size),
                            dtype=np.float32)
        mask_agg = np.zeros((NUM_PAF_CHANNELS, self.heatmap_size, self.heatmap_size),
                             dtype=np.float32)

        for ann in self.samples[iid]:
            joints = load_coco_joints(ann, self.input_size, orig_w, orig_h)
            # Scale joint coordinates to heatmap space
            scaled = [(x * self.scale, y * self.scale, v) for x, y, v in joints]

            # Heatmap: element-wise max across persons
            hm = make_hrnet_heatmap(scaled, self.heatmap_size, self.sigma)
            np.maximum(hm_agg, hm, out=hm_agg)

            # PAF: for overlapping limbs, keep the one with larger magnitude
            paf, pmask = make_hrnet_paf(scaled, self.heatmap_size, self.paf_sigma)
            for limb_idx in range(NUM_COCO_LIMBS):
                paf_limb_mag = np.sqrt(paf[limb_idx * 2] ** 2 + paf[limb_idx * 2 + 1] ** 2)
                agg_limb_mag = np.sqrt(paf_agg[limb_idx * 2] ** 2 + paf_agg[limb_idx * 2 + 1] ** 2)
                update_mask = paf_limb_mag > agg_limb_mag
                paf_agg[limb_idx * 2][update_mask] = paf[limb_idx * 2][update_mask]
                paf_agg[limb_idx * 2 + 1][update_mask] = paf[limb_idx * 2 + 1][update_mask]

            np.maximum(mask_agg, pmask, out=mask_agg)

        paf_t = torch.from_numpy(paf_agg)
        hm_t = torch.from_numpy(hm_agg)
        mask_t = torch.from_numpy(mask_agg)

        return img_tensor, paf_t, hm_t, mask_t
