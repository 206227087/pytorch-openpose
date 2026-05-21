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
from config import NUM_JOINTS, NUM_PAF_CHANNELS, SKELETONS, NUM_LIMBS, KEYPOINT_FLIP_MAP


def make_heatmap(joints, size, sigma, y_grid, x_grid):
    """Generate Gaussian heatmaps for 18 key points.

    Args:
        joints: list of (x, y, visibility) tuples.
        size: heatmap spatial size (e.g., 64 for 256x256 input with stride 4).
        sigma: Gaussian spread in heatmap-space pixels.
        y_grid, x_grid: pre-computed meshgrid arrays.

    Returns:
        hm: (NUM_JOINTS, size, size) float32 array.
    """
    hm = np.zeros((NUM_JOINTS, size, size), dtype=np.float32)

    for j_idx, (x, y, v) in enumerate(joints):
        if v == 0 or j_idx >= NUM_JOINTS:
            continue
        g = np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma ** 2))
        np.maximum(hm[j_idx], g, out=hm[j_idx])

    return hm


def make_paf(joints, size, sigma, y_grid, x_grid):
    """Generate Part Affinity Fields for NUM_LIMBS (18 limbs 36 channels).

    Args:
        joints: list of (x, y, visibility) tuples.
        size: PAF spatial size (e.g., 64).
        sigma: PAF limb width in heatmap-space pixels.
        y_grid, x_grid: pre-computed meshgrid arrays.

    Returns:
        paf: (NUM_PAF_CHANNELS, size, size) float32 array.
        mask: (NUM_PAF_CHANNELS, size, size) float32 array (1 where limb is present).
    """
    paf = np.zeros((NUM_PAF_CHANNELS, size, size), dtype=np.float32)
    mask = np.zeros((NUM_PAF_CHANNELS, size, size), dtype=np.float32)

    for limb_idx, (ja, jb) in enumerate(SKELETONS):
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


def load_joints(ann, input_size, orig_w, orig_h):
    """Load 17 COCO key points from annotation, scaled to input_size.
    Calculate neck point by left shoulder and right shoulder.

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
    for i in range(NUM_JOINTS - 1):
        x, y, v = kps[i * 3], kps[i * 3 + 1], kps[i * 3 + 2]
        if v > 0:
            x = x * sx
            y = y * sy
            x = min(max(x, 0), input_size - 1)
            y = min(max(y, 0), input_size - 1)
        joints.append((x, y, v))

    # Calculate neck as midpoint of left shoulder (index 5) and right shoulder (index 6)
    left_shoulder = joints[5]  # (x, y, v)
    right_shoulder = joints[6]  # (x, y, v)

    # Neck visibility: both shoulders must be visible
    neck_v = min(left_shoulder[2], right_shoulder[2])

    if neck_v > 0:
        neck_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
        neck_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
        neck_x = min(max(neck_x, 0), input_size - 1)
        neck_y = min(max(neck_y, 0), input_size - 1)
    else:
        neck_x = 0.0
        neck_y = 0.0

    joints.append((neck_x, neck_y, neck_v))
    return joints


class HRNetCocoDataset(Dataset):
    """COCO keypoint dataset for HRNet multi-person training.

    Generates 18-channel heatmap targets AND 36-channel PAF targets with
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
        augment: whether to apply data augmentation (default True for train).
    """

    def __init__(self, data_dir, split="train2017",
                 input_size=256, heatmap_size=64, sigma=2.0, paf_sigma=2.0,
                 augment=True, filter_key_points_nums=0):
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.paf_sigma = paf_sigma
        self.scale = heatmap_size / input_size
        self.augment = augment

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
            if ann.get("num_keypoints", 0) <= filter_key_points_nums:
                continue
            iid = ann["image_id"]
            if iid not in self.samples:
                self.samples[iid] = []
            self.samples[iid].append(ann)

        self.image_ids = sorted(self.samples.keys())
        print(f"Loaded {len(self.image_ids)} images with keypoints for {split}")
        if self.augment:
            print(f"  Data augmentation enabled: random horizontal flip, scale, rotation, color jitter")

        # Pre-compute mgrid for heatmap/PAF generation (size is fixed)
        self.y_grid, self.x_grid = np.mgrid[0:heatmap_size, 0:heatmap_size]

    def __len__(self):
        return len(self.image_ids)

    @staticmethod
    def _flip_joints(joints, width):
        """Flip joint coordinates horizontally and swap left-right key points.

        Args:
            joints: list of (x, y, visibility) tuples.
            width: image width for coordinate flipping.

        Returns:
            Flipped joints list with left-right key point swapping.
        """
        # Step 1: 先交换左右关键点的通道索引
        # 例如：left_ankle (15) 和 right_ankle (16) 交换
        swapped = [None] * len(joints)
        for orig_idx, flip_idx in enumerate(KEYPOINT_FLIP_MAP):
            swapped[orig_idx] = joints[flip_idx]

        # Step 2: 再翻转 x 坐标
        # 此时 swapped[15] 已经是原来的 right_ankle，翻转后变成新的 left_ankle
        flipped = []
        for x, y, v in swapped:
            new_x = width - 1 - x if v > 0 else x
            flipped.append((new_x, y, v))

        return flipped

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

        # Random horizontal flip for training
        do_flip = False
        if self.augment:
            do_flip = np.random.random() < 0.5

        if do_flip:
            img = cv2.flip(img, 1)  # 1 = horizontal flip

        # Data augmentation: scale, rotation, color jitter
        scale_factor = 1.0
        rotation_angle = 0.0
        if self.augment:
            # Random scale: [0.8, 1.2]
            scale_factor = np.random.uniform(0.8, 1.2)
            # Random rotation: [-15, 15] degrees
            rotation_angle = np.random.uniform(-15, 15)

            # Apply scale + rotation + center crop to input_size
            if scale_factor != 1.0 or rotation_angle != 0.0:
                center = (self.input_size / 2, self.input_size / 2)
                M = cv2.getRotationMatrix2D(center, rotation_angle, scale_factor)
                img = cv2.warpAffine(img, M, (self.input_size, self.input_size),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # Color jitter: brightness, contrast, saturation
            if np.random.random() < 0.5:
                # Convert to float for color ops
                img_f = img.astype(np.float32)
                # Brightness: [0.8, 1.2]
                alpha = np.random.uniform(0.8, 1.2)
                img_f *= alpha
                # Contrast: [0.8, 1.2]
                beta = np.random.uniform(0.8, 1.2)
                mean = img_f.mean()
                img_f = mean + beta * (img_f - mean)
                img = np.clip(img_f, 0, 255).astype(np.uint8)

        # Normalize
        img_float = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1))

        # Generate aggregated heatmap and PAF from all person annotations
        hm_agg = np.zeros((NUM_JOINTS, self.heatmap_size, self.heatmap_size),
                          dtype=np.float32)
        paf_agg = np.zeros((NUM_PAF_CHANNELS, self.heatmap_size, self.heatmap_size),
                           dtype=np.float32)
        mask_agg = np.zeros((NUM_PAF_CHANNELS, self.heatmap_size, self.heatmap_size),
                            dtype=np.float32)

        for ann in self.samples[iid]:
            joints = load_joints(ann, self.input_size, orig_w, orig_h)

            # Apply flip to joints if needed
            if do_flip:
                joints = self._flip_joints(joints, self.input_size)

            # Apply scale + rotation to joints
            if scale_factor != 1.0 or rotation_angle != 0.0:
                center = self.input_size / 2
                angle_rad = np.radians(rotation_angle)
                cos_a = np.cos(angle_rad) * scale_factor
                sin_a = np.sin(angle_rad) * scale_factor
                new_joints = []
                for x, y, v in joints:
                    if v > 0:
                        dx, dy = x - center, y - center
                        nx = cos_a * dx + sin_a * dy + center
                        ny = -sin_a * dx + cos_a * dy + center
                        nx = min(max(nx, 0), self.input_size - 1)
                        ny = min(max(ny, 0), self.input_size - 1)
                        new_joints.append((nx, ny, v))
                    else:
                        new_joints.append((x, y, v))
                joints = new_joints

            # Scale joint coordinates to heatmap space
            scaled = [(x * self.scale, y * self.scale, v) for x, y, v in joints]

            # Heatmap: element-wise max across persons
            hm = make_heatmap(scaled, self.heatmap_size, self.sigma, self.y_grid, self.x_grid)

            np.maximum(hm_agg, hm, out=hm_agg)

            # PAF: for overlapping limbs, keep the one with larger magnitude
            paf, pmask = make_paf(scaled, self.heatmap_size, self.paf_sigma, self.y_grid, self.x_grid)

            for limb_idx in range(NUM_LIMBS):
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
