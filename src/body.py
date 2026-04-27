"""Body pose estimation inference pipeline.

Detects 18 body keypoints and assembles persons using PAF connections.
"""

import math
import os
import cv2
import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter

from src import preprocessing
from src import util
from src.config import (
    NUM_JOINTS, NUM_LIMBS, LIMBS, INPUT_SIZE, STRIDE,
    SCALE_SEARCH, PEAK_THRESHOLD, PAF_SCORE_THRESHOLD, MID_NUM,
)
from src.model import bodypose_model
from src.preprocessing import normalize_image


def visualize_heatmap(oriImg, heatmap_avg):
    """Visualize heatmap on original image.

    Args:
        oriImg: Original image (H, W, 3) in BGR
        heatmap_avg: (H, W, 19) heatmap array

    Returns:
        vis_img: Image with heatmap overlay
    """
    vis_img = oriImg.copy()

    # Create a colormap for visualization
    colors = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
        [255, 0, 255], [0, 255, 255], [128, 0, 0], [0, 128, 0],
        [0, 0, 128], [128, 128, 0], [128, 0, 128], [0, 128, 128],
        [255, 128, 0], [255, 0, 128], [128, 255, 0], [0, 255, 128],
        [128, 0, 255], [0, 128, 255]
    ]

    # Overlay each joint's heatmap
    alpha = 0.5
    for part in range(NUM_JOINTS):
        heatmap = heatmap_avg[:, :, part]
        heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = np.zeros((oriImg.shape[0], oriImg.shape[1], 3), dtype=np.uint8)

        color = colors[part % len(colors)]
        for i in range(3):
            heatmap_colored[:, :, i] = heatmap_normalized.astype(np.uint8) * color[i] / 255.0

        mask = heatmap_normalized > 50
        vis_img[mask] = vis_img[mask] * (1 - alpha) + heatmap_colored[mask] * alpha

    return vis_img


def visualize_paf(oriImg, paf_avg, limb_idx=None):
    """Visualize PAF on original image.

    Args:
        oriImg: Original image (H, W, 3) in BGR
        paf_avg: (H, W, 38) PAF array
        limb_idx: Specific limb index to visualize (None for all)

    Returns:
        vis_img: Image with PAF visualization
    """
    vis_img = oriImg.copy()

    if limb_idx is not None:
        # Visualize specific limb
        limbs_to_show = [limb_idx]
    else:
        # Visualize first few limbs for clarity
        limbs_to_show = range(min(5, NUM_LIMBS))

    colors = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
        [255, 0, 255]
    ]

    step = 20  # Downsample for arrow visualization

    for limb_k in limbs_to_show:
        paf_x = paf_avg[:, :, limb_k * 2]
        paf_y = paf_avg[:, :, limb_k * 2 + 1]

        # Calculate magnitude
        magnitude = np.sqrt(paf_x ** 2 + paf_y ** 2)

        # Sample points for arrows
        for y in range(0, oriImg.shape[0], step):
            for x in range(0, oriImg.shape[1], step):
                mag = magnitude[y, x]
                if mag > 0.1:  # Only show significant vectors
                    dx = paf_x[y, x] * 10
                    dy = paf_y[y, x] * 10

                    color = colors[limb_k % len(colors)]
                    cv2.arrowedLine(vis_img,
                                    (x, y),
                                    (int(x + dx), int(y + dy)),
                                    color, 1, tipLength=0.3)

    return vis_img


class Body:
    """Multi-person body pose estimator.

    Usage:
        body = Body('model/body_pose_model.pth')
        candidate, subset = body(image)
    """

    def __init__(self, model_path):
        self.model = bodypose_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        state = torch.load(model_path, map_location=lambda s, l: s)
        try:
            self.model.load_state_dict(state)
        except RuntimeError:
            # Handle Caffe-converted weights with different key naming
            model_dict = util.transfer(self.model, state)
            self.model.load_state_dict(model_dict)
        self.model.eval()

    def __call__(self, oriImg):
        """Run body pose estimation on an image.

        Args:
            oriImg: Input image (H, W, 3) in BGR order

        Returns:
            candidate: (N, 4) array of [x, y, score, id] for each detected keypoint
            subset: (M, 20) array where each row is a person:
                    M 检测到的人数
                    columns 0-17 = keypoint indices into candidate,
                    column 18 = total score,
                    column 19 = total number of detected parts
        """

        # 多尺度预测融合
        # 对图像进行多个尺度的缩放（通过 SCALE_SEARCH 配置）
        # 调整大小并填充到模型输入要求
        # 获取 PAF 输出和 Heatmap 输出
        # 将输出恢复到原始图像尺寸
        # 累加平均，融合多个尺度的结果
        # heatmap_avg: (H, W, 19) - 18 个关键点的置信度图 + 背景
        # paf_avg: (H, W, 38) - 19 个肢体的方向场（每个肢体 2 通道：x 和 y 分量）
        # multiplier = [s * INPUT_SIZE / oriImg.shape[0] for s in SCALE_SEARCH]
        # heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
        # paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

        # for scale in multiplier:
        #     data, padded_shape, pad = preprocessing.prepare_model_input(
        #         oriImg, scale, STRIDE)
        #
        #     with torch.no_grad():
        #         paf_output, hm_output = self.model(data)
        #         paf_output = paf_output.cpu().numpy()
        #         hm_output = hm_output.cpu().numpy()
        #
        #     # Extract and resize PAFs
        #     paf = np.transpose(np.squeeze(paf_output), (1, 2, 0))
        #     paf = preprocessing.resize_output(paf, STRIDE, padded_shape, pad, oriImg.shape)
        #     paf_avg += paf / len(multiplier)
        #
        #     # Extract and resize heatmaps
        #     heatmap = np.transpose(np.squeeze(hm_output), (1, 2, 0))
        #     heatmap = preprocessing.resize_output(heatmap, STRIDE, padded_shape, pad, oriImg.shape)
        #     heatmap_avg += heatmap / len(multiplier)

        img = cv2.resize(oriImg, (INPUT_SIZE, INPUT_SIZE))
        img = normalize_image(img)

        tensor = np.transpose(img[:, :, :, np.newaxis], (3, 2, 0, 1))
        tensor = np.ascontiguousarray(tensor)
        data = torch.from_numpy(tensor).float()
        if torch.cuda.is_available():
            data = data.cuda()

        with torch.no_grad():
            paf_output, hm_output = self.model(data)
            paf_output = paf_output.cpu().numpy()
            heatmap_out = hm_output.cpu().numpy()

        paf_avg = np.transpose(np.squeeze(paf_output), (1, 2, 0))
        heatmap_avg = np.transpose(np.squeeze(heatmap_out), (1, 2, 0))

        # Resize 回原始图像尺寸
        paf_avg = cv2.resize(paf_avg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap_avg = cv2.resize(heatmap_avg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        # 在原图上绘制heatmap_avg和paf_avg，并展示
        save_dir = 'output/body_pose'
        os.makedirs(save_dir, exist_ok=True)

        # Visualize heatmap
        heatmap_vis = visualize_heatmap(oriImg, heatmap_avg)
        heatmap_path = os.path.join(save_dir, 'heatmap_overlay.jpg')
        cv2.imwrite(heatmap_path, heatmap_vis)
        print(f"Heatmap visualization saved to: {heatmap_path}")

        # Visualize PAF (first 5 limbs)
        paf_vis = visualize_paf(oriImg, paf_avg)
        paf_path = os.path.join(save_dir, 'paf_overlay.jpg')
        cv2.imwrite(paf_path, paf_vis)
        print(f"PAF visualization saved to: {paf_path}")

        for part in range(NUM_JOINTS):
            heatmap_single = heatmap_avg[:, :, part]
            heatmap_single = gaussian_filter(heatmap_single, sigma=3)
            heatmap_normalized = cv2.normalize(heatmap_single, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)

            # Blend with original image
            blend = cv2.addWeighted(oriImg, 0.6, heatmap_colored, 0.4, 0)
            single_path = os.path.join(save_dir, f'heatmap_joint_{part:02d}.jpg')
            cv2.imwrite(single_path, blend)

        # --- Peak detection （峰值检测）---
        # 对18个关键点类型进行
        # 1、高斯平滑：减少噪声
        # 2、非极大值抑制，找出局部最大值（比较上下左右四个方向的值，必须大于阈值 PEAK_THRESHOLD）
        # 3、记录峰值，保存位置 (x, y)、置信度分数、全局 ID
        all_peaks = []
        peak_counter = 0

        for part in range(NUM_JOINTS):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=3)

            # Find local maxima
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
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            peak_id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],)
                                       for i in range(len(peak_id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        # --- PAF connection scoring （PAF 连接评分）---
        connection_all = []
        special_k = []

        # 遍历 19 个肢体类型
        for k in range(NUM_LIMBS):
            # 获取肢体连接的两个关键点索引
            ja, jb = LIMBS[k]
            # PAF channels for this limb: k*2 (x-component), k*2+1 (y-component)
            score_mid = paf_avg[:, :, [k * 2, k * 2 + 1]]
            candA = all_peaks[ja]
            candB = all_peaks[jb]
            nA = len(candA)
            nB = len(candB)

            if nA == 0 or nB == 0:
                special_k.append(k)
                connection_all.append([])
            else:
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        # 计算方向向量：从 A 指向 B 的单位向量
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        norm = max(0.001, norm)
                        vec = np.divide(vec, norm)

                        startend = list(zip(
                            np.linspace(candA[i][0], candB[j][0], num=MID_NUM),
                            np.linspace(candA[i][1], candB[j][1], num=MID_NUM),
                        ))

                        vec_x = np.array([
                            score_mid[int(round(startend[I][1])),
                            int(round(startend[I][0])), 0]
                            for I in range(len(startend))
                        ])
                        vec_y = np.array([
                            score_mid[int(round(startend[I][1])),
                            int(round(startend[I][0])), 1]
                            for I in range(len(startend))
                        ])
                        # 采样中间点：在 A-B 连线上均匀采样 MID_NUM 个点
                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = (
                                sum(score_midpts) / len(score_midpts)
                                + min(0.5 * oriImg.shape[0] / norm - 1, 0)
                        )
                        # AF 一致性检查,criterion1: 超过 80% 的采样点得分 > 阈值
                        criterion1 = len(np.nonzero(score_midpts > PAF_SCORE_THRESHOLD)[0]) > 0.8 * len(score_midpts)
                        # 平均得分考虑距离惩罚后仍为正
                        criterion2 = score_with_dist_prior > 0

                        if criterion1 and criterion2:
                            connection_candidate.append([
                                i, j, score_with_dist_prior,
                                score_with_dist_prior + candA[i][2] + candB[j][2]
                            ])

                # 贪心分配：按分数排序，确保每个关键点最多被分配一次
                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if i not in connection[:, 0] and j not in connection[:, 1]:
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, candA[i][2], candB[j][2]]])
                    if len(connection) >= min(nA, nB):
                        break

                connection_all.append(connection)

        # --- Person assembly （人员组装，将独立的肢体连接组合成完整的人）---
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        # 遍历所有肢体连
        for k in range(NUM_LIMBS):
            if k in special_k:
                continue
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = LIMBS[k]

            # 对于每个连接 (partA, partB)：
            # 情况 1：只找到一个已有的人员包含 partA 或 partB → 将该连接加入此人
            # 情况 2：找到两个不同的人员 → 合并这两个人（如果没有冲突）
            # 情况 3：都没找到 → 创建新的人员条目
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
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
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
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

        # 删除不符合条件的人员:
        # 检测到的关键点数量 < 4
        # 平均置信度 < 0.4
        delete_idx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                delete_idx.append(i)
        subset = np.delete(subset, delete_idx, axis=0)

        return candidate, subset
