"""HRNet-based multi-person body pose estimation inference pipeline.

Follows the same structure as body.py (OpenPose Body class) but uses the
HRNet backbone for heatmap prediction. Since HRNet has no PAF branch,
person grouping is done via skeleton-based affinity scoring instead of
PAF integration.

Returns the same (candidate, subset) format as body.py for compatibility
with util.draw_bodypose() and util.handDetect().

Usage:
    from src.hrnet_body_pose import BodyHRNetPose
    body = BodyHRNetPose('model/hrnet_w32.pth')
    candidate, subset = body(image)
"""

import math
import os

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from src.config import NUM_JOINTS,NUM_LIMBS,SKELETONS
from src.models.hrnet_model import HRNet


# ─── COCO-to-OpenPose Joint Mapping ──────────────────────────────────────────
# Maps COCO 17-joint indices to OpenPose 18-joint indices for compatibility
# with util.draw_bodypose() and util.handDetect().
# OpenPose joints: 0:nose 1:neck 2:r_shoulder 3:r_elbow 4:r_wrist
#                  5:l_shoulder 6:l_elbow 7:l_wrist 8:r_hip 9:r_knee
#                 10:r_ankle 11:l_hip 12:l_knee 13:l_ankle
#                 14:r_eye 15:l_eye 16:r_ear 17:l_ear
COCO_TO_OPENPOSE = {
    0: 0,   # nose -> nose
    1: 15,  # left_eye -> l_eye
    2: 14,  # right_eye -> r_eye
    3: 17,  # left_ear -> l_ear
    4: 16,  # right_ear -> r_ear
    5: 5,   # left_shoulder -> l_shoulder
    6: 2,   # right_shoulder -> r_shoulder
    7: 6,   # left_elbow -> l_elbow
    8: 3,   # right_elbow -> r_elbow
    9: 7,   # left_wrist -> l_wrist
    10: 4,  # right_wrist -> r_wrist
    11: 11, # left_hip -> l_hip
    12: 8,  # right_hip -> r_hip
    13: 12, # left_knee -> l_knee
    14: 9,  # right_knee -> r_knee
    15: 13, # left_ankle -> l_ankle
    16: 10, # right_ankle -> r_ankle
    17: 1,  # neck -> neck
}

# ─── Inference Parameters ─────────────────────────────────────────────────────
INPUT_SIZE = 256          # HRNet input image size
HEATMAP_SIZE = 64         # Output heatmap size (stride = 4)
STRIDE = INPUT_SIZE // HEATMAP_SIZE

SCALE_SEARCH = [0.5, 1.0, 1.5, 2.0]  # Multi-scale inference scales
PEAK_THRESHOLD = 0.1                  # Heatmap peak detection threshold
MIN_KEYPOINTS = 3                     # Minimum keypoints per person
MIN_AVG_CONF = 0.2                    # Minimum average confidence per person


# ─── Visualization Helpers ────────────────────────────────────────────────────

def visualize_heatmap(oriImg, heatmap_avg):
    """Overlay heatmap on original image for debugging."""
    vis_img = oriImg.copy().astype(np.float64)
    alpha = 0.4
    colors = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
        [0, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128],
        [128, 128, 0], [128, 0, 128], [0, 128, 128],
        [64, 64, 64], [192, 192, 192], [128, 64, 0], [64, 128, 0], [0, 64, 128],
    ]

    for part in range(heatmap_avg.shape[2]):
        heatmap = heatmap_avg[:, :, part]
        heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = np.zeros((oriImg.shape[0], oriImg.shape[1], 3), dtype=np.uint8)
        color = colors[part % len(colors)]
        for i in range(3):
            heatmap_colored[:, :, i] = heatmap_normalized.astype(np.uint8) * color[i] / 255.0
        mask = heatmap_normalized > 50
        vis_img[mask] = vis_img[mask] * (1 - alpha) + heatmap_colored[mask] * alpha

    return vis_img.astype(np.uint8)


# ─── Multi-Person Grouping (skeleton-based, no PAF) ───────────────────────────

def group_keypoints_by_paf(all_peaks, paf_avg, oriImg_shape, mid_num=10,
                            paf_score_threshold=0.05):
    """Group detected keypoints into persons using PAF connection scoring.

    Same algorithm as body.py's PAF scoring: for each limb, integrate
    the PAF along the line segment between candidate keypoint pairs,
    then greedily assign the highest-scoring connections.

    Args:
        all_peaks: list of 17 lists, each containing (x, y, score, id) tuples.
        paf_avg: (H, W, 32) PAF array resized to original image size.
        oriImg_shape: (height, width) of original image.
        mid_num: number of sample points along each PAF for scoring (default 10).
        paf_score_threshold: minimum average PAF score for a valid connection.

    Returns:
        connection_all: list of connection arrays for each limb.
    """
    connection_all = []

    for limb_idx, (indexA, indexB) in enumerate(SKELETONS):
        candA = all_peaks[indexA]
        candB = all_peaks[indexB]
        nA = len(candA)
        nB = len(candB)

        if nA == 0 or nB == 0:
            connection_all.append(np.zeros((0, 5)))
            continue

        connection_candidate = []

        for i in range(nA):
            for j in range(nB):
                x1, y1, scoreA = candA[i][0], candA[i][1], candA[i][2]
                x2, y2, scoreB = candB[j][0], candB[j][1], candB[j][2]

                dx = x2 - x1
                dy = y2 - y1
                dist = math.sqrt(dx ** 2 + dy ** 2)
                if dist < 1e-6:
                    continue

                # Unit vector along the limb
                ux = dx / dist
                uy = dy / dist

                # Sample points along the PAF and compute line integral
                paf_x = paf_avg[:, :, limb_idx * 2]
                paf_y = paf_avg[:, :, limb_idx * 2 + 1]

                score = 0.0 # 沿连线方向采样
                for t in range(mid_num):
                    frac = t / mid_num
                    sx = int(round(x1 + frac * dx))
                    sy = int(round(y1 + frac * dy))
                    sx = min(max(sx, 0), oriImg_shape[1] - 1)
                    sy = min(max(sy, 0), oriImg_shape[0] - 1)
                    score += paf_x[sy, sx] * ux + paf_y[sy, sx] * uy    # 向量点积

                score /= mid_num

                # Filter by threshold
                if score < paf_score_threshold:
                    continue

                connection_candidate.append([i, j, score])

        # Greedy assignment: sort by score, assign each keypoint at most once
        connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
        connection = np.zeros((0, 5))
        for c in range(len(connection_candidate)):  # connection结构：col 0: 关节A的关键点ID（全局ID ）col 1: 关节B的关键点ID（全局ID）col 2: PAF连接得分 col 3: 关节A的关键点置信度 col 4: 关节B的关键点置信度
            i, j, s = connection_candidate[c][0:3]  # 提取关节点索引和paf得分
            if i not in connection[:, 0] and j not in connection[:, 1]:
                connection = np.vstack([
                    connection,
                    [candA[i][3], candB[j][3], s, candA[i][2], candB[j][2]]
                ])
            if len(connection) >= min(nA, nB):
                break

        connection_all.append(connection)

    return connection_all


def assemble_persons(all_peaks, connection_all):
    """Assemble persons from limb connections.

    Follows the same logic as body.py's person assembly:
    - For each connection (partA, partB):
      Case 1: One existing person contains partA or partB -> add connection
      Case 2: Two different persons -> merge if no conflict
      Case 3: Neither found -> create new person

    Returns (candidate, subset) in the same format as body.py:
      candidate: (N, 4) array of [x, y, score, id]
      subset: (M, 19) array - first 17 cols are keypoint IDs, col 17 is
              total score, col 18 is keypoint count
    """
    # subset: NUM_JOINTS keypoint slots + score + count
    num_cols = NUM_JOINTS + 2
    subset = -1 * np.ones((0, num_cols))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(NUM_LIMBS):
        if k >= len(connection_all) or len(connection_all[k]) == 0:
            continue

        indexA, indexB = SKELETONS[k]
        partAs = connection_all[k][:, 0]
        partBs = connection_all[k][:, 1]

        for i in range(len(connection_all[k])):
            found = 0
            subset_idx = [-1, -1]
            for j in range(len(subset)):
                if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                    if found < 2:
                        subset_idx[found] = j
                        found += 1
                    else:
                        break

            if found == 1:
                j = subset_idx[0]
                if subset[j][indexB] != partBs[i]:
                    subset[j][indexB] = partBs[i]
                    subset[j][-1] += 1  # keypoint count
                    subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]  # score
            elif found == 2:
                j1, j2 = subset_idx
                membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                if len(np.nonzero(membership == 2)[0]) == 0:
                    subset[j1][:-2] += (subset[j2][:-2] + 1)
                    subset[j1][-2:] += subset[j2][-2:]
                    subset[j1][-2] += connection_all[k][i][2]
                    subset = np.delete(subset, j2, 0)
                else:
                    subset[j1][indexB] = partBs[i]
                    subset[j1][-1] += 1
                    subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
            elif not found:
                row = -1 * np.ones(num_cols)
                row[indexA] = partAs[i]
                row[indexB] = partBs[i]
                row[-1] = 2  # keypoint count
                row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                subset = np.vstack([subset, row])

    # Filter invalid persons
    delete_idx = []
    for i in range(len(subset)):
        if subset[i][-1] < MIN_KEYPOINTS or subset[i][-2] / subset[i][-1] < MIN_AVG_CONF:
            delete_idx.append(i)
    subset = np.delete(subset, delete_idx, axis=0)

    return candidate, subset


# ─── Format Conversion ────────────────────────────────────────────────────────

def convert_to_openpose_format(candidate, subset):
    """Convert 18-joint (candidate, subset) to OpenPose 18-joint format.

    Training uses 18 joints (COCO 17 + neck), so neck is already at index 17.
    This function reorders to OpenPose's joint ordering (neck at index 1).

    Args:
        candidate: (N, 4) array of [x, y, score, id] from BodyHRNetPose.
        subset: (M, 20) array from BodyHRNetPose (18 joints + score + count).

    Returns:
        candidate: same candidate array (keypoints are unchanged).
        subset_op: (M, 20) array in OpenPose format:
            - cols 0-17: keypoint indices (18 OpenPose joints)
            - col 18: total score
            - col 19: keypoint count
    """
    if len(subset) == 0:
        return candidate, -1 * np.ones((0, 20))

    subset_op = -1 * np.ones((len(subset), 20))

    for i in range(len(subset)):
        # Map joint positions to OpenPose ordering
        for coco_idx, openpose_idx in COCO_TO_OPENPOSE.items():
            if coco_idx < NUM_JOINTS:
                subset_op[i, openpose_idx] = subset[i, coco_idx]

        # Copy score and count
        subset_op[i, 18] = subset[i, NUM_JOINTS]      # total score
        subset_op[i, 19] = subset[i, NUM_JOINTS + 1]   # keypoint count

    return candidate, subset_op


# ─── BodyHRNetPose Class ─────────────────────────────────────────────────────

class BodyHRNetPose:
    """Multi-person body pose estimator using HRNet backbone.

    Follows the same interface as body.py's Body class:
        body = BodyHRNetPose('model/hrnet_w32.pth')
        candidate, subset = body(image)

    Returns:
        candidate: (N, 4) array of [x, y, score, id] for all detected keypoints.
        subset: (M, 20) array where each row is a person:
            - cols 0-17: keypoint index into candidate array (-1 if missing)
            - col 18: sum of keypoint scores + connection scores
            - col 19: number of detected keypoints

    This format is compatible with util.draw_bodypose() and util.handDetect().

    Args:
        model_path: path to HRNet weights file.
        width: HRNet width (32 for W32, 48 for W48).
        input_size: model input image size (default 256).
        scale_search: list of scales for multi-scale inference.
    """

    def __init__(self, model_path, width=32, input_size=256,
                 scale_search=None):
        self.input_size = input_size
        self.width = width
        self.scale_search = scale_search or SCALE_SEARCH
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build and load model (dual-branch: PAF + heatmap)
        self.model = HRNet(num_joints=NUM_JOINTS, num_limbs=NUM_LIMBS, width=width)
        state = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']

        # Handle legacy checkpoints with 'final_layer' instead of 'heatmap_head'/'paf_head'
        # if 'final_layer.weight' in state and 'heatmap_head.0.weight' not in state:
        #     print(f"  Converting legacy checkpoint (final_layer -> heatmap_head + paf_head)")
        #     final_weight = state.pop('final_layer.weight')
        #     final_bias = state.pop('final_layer.bias')
        #
        #     # Current head: Conv3x3->BN->ReLU->Conv3x3->BN->ReLU->Dropout->Conv3x3->BN->ReLU->Conv1x1
        #     # Final Conv1x1 is at index 10
        #     state['heatmap_head.10.weight'] = final_weight
        #     state['heatmap_head.10.bias'] = final_bias
        #
        #     # PAF head doesn't exist in legacy checkpoint - leave it randomly initialized
        #     print(f"  Warning: legacy checkpoint has no PAF head. PAF connections will be random.")
        #     print(f"  For best results, use a checkpoint trained with the dual-branch architecture.")

        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()
        print(f"BodyHRNetPose: HRNet-W{width} loaded from {model_path}")

    def __call__(self, oriImg):
        """Run multi-person body pose estimation on an image.

        Pipeline (mirrors body.py):
        1. Multi-scale inference -> average heatmaps
        2. Resize heatmaps to original image size
        3. Peak detection (Gaussian smooth + NMS)
        4. Skeleton-based connection scoring (replaces PAF scoring)
        5. Person assembly (same greedy logic as body.py)
        6. Filter invalid persons

        Args:
            oriImg: Input image (H, W, 3) in BGR order.

        Returns:
            candidate: (N, 4) array of [x, y, score, id].
            subset: (M, 20) array of person data.
        """
        h, w = oriImg.shape[0:2]

        img = cv2.resize(oriImg, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                paf_output, hm_output = self.model(img)
        paf_np = paf_output.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.float32)  # (H, W, C)
        heatmap_np = hm_output.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.float32)

        # 【可选】在64x64上做粗略阈值过滤
        heatmap_np[heatmap_np < 0.05] = 0  # 提前过滤噪声

        # Resize to scaled image size, then to original size
        paf_output = cv2.resize(paf_np, (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap_output = cv2.resize(heatmap_np, (w, h), interpolation=cv2.INTER_CUBIC)

        # ── Debug: save heatmap visualization ──
        save_dir = '../output/hrnet_body_pose'
        os.makedirs(save_dir, exist_ok=True)
        heatmap_vis = visualize_heatmap(oriImg, heatmap_output)
        cv2.imwrite(os.path.join(save_dir, 'heatmap_overlay.jpg'), heatmap_vis)

        # Save per-joint heatmaps
        for part in range(NUM_JOINTS):
            hm_single = heatmap_output[:, :, part].astype(np.float32)
            hm_norm = cv2.normalize(hm_single, None, 0, 255, cv2.NORM_MINMAX)
            hm_colored = cv2.applyColorMap(hm_norm.astype(np.uint8), cv2.COLORMAP_JET)
            blend = cv2.addWeighted(oriImg, 0.6, hm_colored, 0.4, 0)
            cv2.imwrite(os.path.join(save_dir, f'heatmap_joint_{part:02d}.jpg'), blend)

        # ── Step 2: Peak detection ──
        # Same logic as body.py: Gaussian smooth + NMS
        all_peaks = []
        peak_counter = 0

        for part in range(NUM_JOINTS):
            map_ori = heatmap_output[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=3)

            # Non-maximum suppression: compare with 4 neighbors
            map_left = np.zeros(one_heatmap.shape)
            map_left[1:, :] = one_heatmap[:-1, :]
            map_right = np.zeros(one_heatmap.shape)
            map_right[:-1, :] = one_heatmap[1:, :]
            map_up = np.zeros(one_heatmap.shape)
            map_up[:, 1:] = one_heatmap[:, :-1]
            map_down = np.zeros(one_heatmap.shape)
            map_down[:, :-1] = one_heatmap[:, 1:]

            peaks_binary = np.logical_and.reduce((
                one_heatmap >= map_left,
                one_heatmap >= map_right,
                one_heatmap >= map_up,
                one_heatmap >= map_down,
                one_heatmap > PEAK_THRESHOLD,
            ))
            # 查找每个维度上非零（True）元素的索引（y,x）坐标，通过zip转换为（x,y）
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
            #  map_ori 是NumPy数组，需要用 [行, 列] 即，(y,x)，将score加入到peaks，形成（x,y,score）
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            peak_id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],)
                                       for i in range(len(peak_id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        # ── Step 3: PAF connection scoring ──
        connection_all = group_keypoints_by_paf(
            all_peaks, paf_output, (h, w)
        )

        # ── Step 4: Person assembly ──
        candidate, subset = assemble_persons(all_peaks, connection_all)

        return candidate, subset
