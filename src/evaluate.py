"""Evaluation metrics for pose estimation models.

Provides PCK (Percentage of Correct Keypoints) and PCKh (normalized by
head size) metrics, as well as OKS-based mAP for COCO evaluation.
"""

import numpy as np
import torch


# ─── PCK: Percentage of Correct Keypoints ─────────────────────────────────────

def compute_pck(pred_keypoints, gt_keypoints, threshold=0.5,
                normalize='torso', gt_normalizer=None):
    """Compute PCK (Percentage of Correct Keypoints).

    A predicted keypoint is considered correct if its distance from the
    ground truth is within a threshold * normalize_size.

    Args:
        pred_keypoints: (N, K, 2) predicted keypoint coordinates [x, y].
        gt_keypoints: (N, K, 2) ground truth keypoint coordinates [x, y].
        threshold: PCK threshold (default 0.5 for PCK@0.5).
        normalize: normalization method:
            'torso': use torso size (left_shoulder to right_hip distance)
            'head': use head size (PCKh)
            'bbox': use bounding box diagonal
            'custom': use gt_normalizer array
        gt_normalizer: (N,) array of per-sample normalization sizes
            (required when normalize='custom').

    Returns:
        pck: float, percentage of correct keypoints.
        per_joint_pck: (K,) array of per-joint PCK.
    """
    pred = np.asarray(pred_keypoints)
    gt = np.asarray(gt_keypoints)
    N, K, _ = pred.shape

    # Compute normalization sizes
    if normalize == 'torso':
        # OpenPose: left_shoulder=5, right_hip=8
        norm_sizes = np.zeros(N)
        for i in range(N):
            ls = gt[i, 5]
            rh = gt[i, 8]
            norm_sizes[i] = np.sqrt((ls[0] - rh[0]) ** 2 + (ls[1] - rh[1]) ** 2)
            if norm_sizes[i] < 1e-6:
                # Fallback: use full bounding box diagonal
                valid = gt[i][np.any(gt[i] > 0, axis=-1)]
                if len(valid) > 1:
                    norm_sizes[i] = np.linalg.norm(valid.max(0) - valid.min(0))
                else:
                    norm_sizes[i] = 1.0
    elif normalize == 'head':
        # PCKh: head size from nose(0) and neck(1) or head bbox
        norm_sizes = np.zeros(N)
        for i in range(N):
            nose = gt[i, 0]
            neck = gt[i, 1]
            norm_sizes[i] = np.sqrt((nose[0] - neck[0]) ** 2 + (nose[1] - neck[1]) ** 2) * 2
            if norm_sizes[i] < 1e-6:
                valid = gt[i][np.any(gt[i] > 0, axis=-1)]
                if len(valid) > 1:
                    norm_sizes[i] = np.linalg.norm(valid.max(0) - valid.min(0))
                else:
                    norm_sizes[i] = 1.0
    elif normalize == 'bbox':
        norm_sizes = np.zeros(N)
        for i in range(N):
            valid = gt[i][np.any(gt[i] > 0, axis=-1)]
            if len(valid) > 1:
                norm_sizes[i] = np.linalg.norm(valid.max(0) - valid.min(0))
            else:
                norm_sizes[i] = 1.0
    elif normalize == 'custom':
        norm_sizes = np.asarray(gt_normalizer)
    else:
        raise ValueError(f"Unknown normalize method: {normalize}")

    # Compute distances
    distances = np.sqrt(np.sum((pred - gt) ** 2, axis=-1))  # (N, K)

    # Check correctness
    correct = distances < threshold * norm_sizes[:, None]  # (N, K)

    # Handle invisible/missing keypoints (marked as 0,0)
    visible = np.any(gt > 0, axis=-1)  # (N, K)
    correct = correct & visible

    # Compute PCK
    total_visible = visible.sum()
    if total_visible == 0:
        return 0.0, np.zeros(K)

    pck = correct.sum() / total_visible
    per_joint_pck = np.zeros(K)
    for k in range(K):
        vis_k = visible[:, k].sum()
        if vis_k > 0:
            per_joint_pck[k] = correct[:, k].sum() / vis_k

    return pck, per_joint_pck


def compute_pckh(pred_keypoints, gt_keypoints, threshold=0.5):
    """Compute PCKh (PCK normalized by head size).

    Convenience wrapper around compute_pck with normalize='head'.

    Args:
        pred_keypoints: (N, K, 2) predicted keypoint coordinates.
        gt_keypoints: (N, K, 2) ground truth keypoint coordinates.
        threshold: PCKh threshold (default 0.5).

    Returns:
        pckh: float, percentage of correct keypoints.
        per_joint_pckh: (K,) array of per-joint PCKh.
    """
    return compute_pck(pred_keypoints, gt_keypoints, threshold=threshold,
                       normalize='head')


# ─── OKS-based mAP (COCO-style) ──────────────────────────────────────────────

# COCO keypoint sigmas (per-joint standard deviations for OKS)
COCO_SIGMAS = np.array([
    0.026,  # nose
    0.025,  # left_eye
    0.025,  # right_eye
    0.035,  # left_ear
    0.035,  # right_ear
    0.079,  # left_shoulder
    0.079,  # right_shoulder
    0.072,  # left_elbow
    0.072,  # right_elbow
    0.062,  # left_wrist
    0.062,  # right_wrist
    0.107,  # left_hip
    0.107,  # right_hip
    0.087,  # left_knee
    0.087,  # right_knee
    0.089,  # left_ankle
    0.089,  # right_ankle
])


def compute_oks(pred_kpts, gt_kpts, area, sigmas=None):
    """Compute Object Keypoint Similarity (OKS).

    Args:
        pred_kpts: (K, 3) predicted keypoints [x, y, score].
        gt_kpts: (K, 3) ground truth keypoints [x, y, visible].
        area: bounding box area of the ground truth person.
        sigmas: per-joint sigmas (default COCO_SIGMAS).

    Returns:
        OKS score in [0, 1].
    """
    if sigmas is None:
        sigmas = COCO_SIGMAS

    K = len(gt_kpts)
    visible = gt_kpts[:, 2] > 0
    if visible.sum() == 0:
        return 0.0

    d = (pred_kpts[:, :2] - gt_kpts[:, :2]) ** 2
    s = 2 * sigmas ** 2
    e = d.sum(axis=1) / (s * area + 1e-8)
    oks = np.exp(-e)
    oks = (oks * visible).sum() / visible.sum()
    return oks


def evaluate_model(model, val_loader, device='cuda', input_size=368):
    """Evaluate a pose estimation model on a validation set.

    Computes PCK@0.5, PCKh@0.5, and per-joint accuracy.

    Args:
        model: nn.Module in eval mode.
        val_loader: DataLoader yielding (image, gt_keypoints) batches.
        device: compute device.
        input_size: model input size.

    Returns:
        Dict with evaluation metrics.
    """
    model.eval()
    all_pred = []
    all_gt = []

    with torch.inference_mode():
        for batch in val_loader:
            images = batch[0].to(device)
            gt_kpts = batch[1].numpy()  # (N, K, 2)

            outputs = model(images)
            if isinstance(outputs, tuple):
                # OpenPose returns (PAF, heatmap)
                heatmaps = outputs[1]
            else:
                heatmaps = outputs

            # Extract keypoints from heatmaps
            pred_kpts = _extract_keypoints_from_heatmaps(heatmaps, input_size)
            all_pred.append(pred_kpts)
            all_gt.append(gt_kpts)

    all_pred = np.concatenate(all_pred, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)

    pck_05, per_joint_pck = compute_pck(all_pred, all_gt, threshold=0.5)
    pckh_05, per_joint_pckh = compute_pckh(all_pred, all_gt, threshold=0.5)
    pck_02, _ = compute_pck(all_pred, all_gt, threshold=0.2)

    return {
        'PCK@0.5': pck_05,
        'PCK@0.2': pck_02,
        'PCKh@0.5': pckh_05,
        'per_joint_PCK': per_joint_pck,
        'per_joint_PCKh': per_joint_pckh,
    }


def _extract_keypoints_from_heatmaps(heatmaps, input_size):
    """Extract keypoint coordinates from heatmaps via argmax.

    Args:
        heatmaps: (N, K, H, W) heatmap tensor.
        input_size: original input image size.

    Returns:
        (N, K, 2) array of [x, y] coordinates.
    """
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.cpu().numpy()

    N, K, H, W = heatmaps.shape
    stride = input_size / H
    keypoints = np.zeros((N, K, 2))

    for n in range(N):
        for k in range(K):
            idx = np.unravel_index(heatmaps[n, k].argmax(), (H, W))
            keypoints[n, k, 1] = idx[0] * stride  # y
            keypoints[n, k, 0] = idx[1] * stride  # x

    return keypoints
