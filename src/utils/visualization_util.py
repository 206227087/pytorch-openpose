"""
@Author: chaos
@Date: 2026/4/29
@Version：V1.0 
@Description：
"""
import os

import cv2
import numpy as np

from src.config import NUM_JOINTS, SKELETONS, JOINT_NAMES

# Build joint-to-limbs mapping: for each joint, find all connected limbs
# joint_to_limbs[joint_idx] = [(limb_idx, is_start_point), ...]
joint_to_limbs = {}
for joint_idx in range(NUM_JOINTS):
    joint_to_limbs[joint_idx] = []

for limb_idx, (j1, j2) in enumerate(SKELETONS):
    joint_to_limbs[j1].append((limb_idx, True))  # limb starts from j1
    joint_to_limbs[j2].append((limb_idx, False))  # limb ends at j2


def save_heatmap_comparison(img, hm_pred, hm_gt, paf_pred, paf_gt, epoch, step, save_dir):
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

    new_imgs = []
    # Part 1: Visualize heatmap by joints
    for j in range(hm_pred.shape[0]):
        joint_name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f'joint_{j}'

        gt_hm = hm_gt[j].astype(np.float32)
        gt_hm = cv2.resize(gt_hm, (img_vis.shape[1], img_vis.shape[0]), interpolation=cv2.INTER_CUBIC)
        gt_hm = cv2.normalize(gt_hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        gt_hm = cv2.applyColorMap(gt_hm, cv2.COLORMAP_JET)
        gt_blend = cv2.addWeighted(img_vis, 0.5, gt_hm, 0.5, 0)
        cv2.putText(gt_blend, f'GT-HM-{joint_name}', (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        pred_hm = hm_pred[j].astype(np.float32)
        pred_hm = cv2.resize(pred_hm, (img_vis.shape[1], img_vis.shape[0]), interpolation=cv2.INTER_CUBIC)
        pred_hm = cv2.normalize(pred_hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        pred_hm = cv2.applyColorMap(pred_hm, cv2.COLORMAP_JET)
        pred_blend = cv2.addWeighted(img_vis, 0.5, pred_hm, 0.5, 0)
        cv2.putText(pred_blend, f'PRED-GM-{joint_name}', (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        comparison = np.concatenate([gt_blend, pred_blend], axis=1)
        new_imgs.append(comparison)

    # Part 2: Visualize PAF by joint (show all limbs connected to each joint)
    for joint_idx in range(NUM_JOINTS):
        connected_limbs = joint_to_limbs[joint_idx]
        if not connected_limbs:
            continue

        # Aggregate PAF magnitude for all limbs connected to this joint
        gt_mag_agg = np.zeros((img_vis.shape[0], img_vis.shape[1]), dtype=np.float32)
        pred_mag_agg = np.zeros((img_vis.shape[0], img_vis.shape[1]), dtype=np.float32)

        for limb_idx, is_start in connected_limbs:
            # GT PAF magnitude for this limb
            gt_px = paf_gt[limb_idx * 2]
            gt_py = paf_gt[limb_idx * 2 + 1]
            gt_mag = np.sqrt(gt_px ** 2 + gt_py ** 2).astype(np.float32)
            gt_mag_resized = cv2.resize(gt_mag, (img_vis.shape[1], img_vis.shape[0]), interpolation=cv2.INTER_CUBIC)
            gt_mag_agg = np.maximum(gt_mag_agg, gt_mag_resized)  # Take max across limbs

            # Pred PAF magnitude for this limb
            pred_px = paf_pred[limb_idx * 2]
            pred_py = paf_pred[limb_idx * 2 + 1]
            pred_mag = np.sqrt(pred_px ** 2 + pred_py ** 2).astype(np.float32)
            pred_mag_resized = cv2.resize(pred_mag, (img_vis.shape[1], img_vis.shape[0]),
                                          interpolation=cv2.INTER_CUBIC)
            pred_mag_agg = np.maximum(pred_mag_agg, pred_mag_resized)  # Take max across limbs

        joint_name = JOINT_NAMES[joint_idx] if joint_idx < len(JOINT_NAMES) else f'joint_{joint_idx}'

        # Normalize and visualize aggregated PAF
        gt_norm = cv2.normalize(gt_mag_agg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        gt_norm = cv2.applyColorMap(gt_norm, cv2.COLORMAP_JET)
        gt_blend = cv2.addWeighted(img_vis, 0.5, gt_norm, 0.5, 0)
        cv2.putText(gt_blend, f'GT-PAF-{joint_name}', (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        pred_norm = cv2.normalize(pred_mag_agg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        pred_norm = cv2.applyColorMap(pred_norm, cv2.COLORMAP_JET)
        pred_blend = cv2.addWeighted(img_vis, 0.5, pred_norm, 0.5, 0)
        cv2.putText(pred_blend, f'PRED-PAF-{joint_name}', (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        comparison = np.concatenate([gt_blend, pred_blend], axis=1)
        new_imgs.append(comparison)

    # Save heatmap and PAF comparisons
    for i in range(len(new_imgs) // 2):
        comparison = np.concatenate([new_imgs[i], new_imgs[len(new_imgs) // 2 + i]], axis=0)
        cv2.imwrite(os.path.join(save_dir, f'epoch{epoch:04d}_step{step:02d}_img{i:02d}.jpg'), comparison)
