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
from scipy.ndimage import gaussian_filter, maximum_filter

from src.config import NUM_JOINTS, NUM_LIMBS, SKELETONS, JOINT_NAMES
from src.models.hrnet_model import HRNet

PEAK_THRESHOLD = 0.25  # Heatmap peak detection threshold
MIN_KEYPOINTS = 8  # Minimum keypoints per person
MIN_DIST_THRESHOLD = 15.0  # Distance threshold for peak merging
HEATMAP_FILTER = 0.5
GAUSSIAN_FILTER_SIGMA = 4
PAF_FILTER_THRESHOLD = 0.5

def group_keypoints_by_paf(all_peaks, paf_avg, oriImg_shape, mid_num=10):
    """Compute PAF scores for all possible limb connections."""
    connection_links = []  # Stores: (limb_idx, score, id_a, id_b, x1, y1, x2, y2)

    for limb_idx, (indexA, indexB) in enumerate(SKELETONS):
        candA = all_peaks[indexA]
        candB = all_peaks[indexB]
        if not candA or not candB: continue

        paf_x = paf_avg[:, :, limb_idx * 2]
        paf_y = paf_avg[:, :, limb_idx * 2 + 1]
        h, w = oriImg_shape

        for i, (x1, y1, scoreA, idA) in enumerate(candA):
            for j, (x2, y2, scoreB, idB) in enumerate(candB):
                dx, dy = x2 - x1, y2 - y1
                dist = math.sqrt(dx ** 2 + dy ** 2)
                if dist < 1e-6: continue

                ux, uy = dx / dist, dy / dist
                paf_score = 0.0
                for t in range(mid_num):
                    frac = t / mid_num
                    sx = min(max(int(round(x1 + frac * dx)), 0), w - 1)
                    sy = min(max(int(round(y1 + frac * dy)), 0), h - 1)
                    paf_score += paf_x[sy, sx] * ux + paf_y[sy, sx] * uy
                paf_score /= mid_num

                if paf_score > PAF_FILTER_THRESHOLD:
                    connection_links.append((limb_idx, paf_score, idA, idB, x1, y1, x2, y2))

    # Sort by PAF score descending for greedy assembly
    return sorted(connection_links, key=lambda x: x[1], reverse=True)


def assemble_persons_simple(all_peaks, connection_links):
    """Greedy assembly using dictionary objects for better debugging."""
    persons = []  # List of dicts: {'joints': {joint_idx: global_id}, 'score': float}
    point_to_person = {}  # Map global_id -> person_index

    # 记录每个人身上的连线历史，用于重组决策：{person_idx: [(id_a, id_b, score), ...]}
    person_links = []

    # 按 PAF 得分从高到低排序，优先处理可信度高的连线
    for limb_idx, paf_score, idA, idB, x1, y1, x2, y2 in connection_links:
        idxA, idxB = SKELETONS[limb_idx]
        if limb_idx == 10:
            print(f"8->10:({x1},{y1})->({x2},{y2})，score:{paf_score}")
        # print(f"线段index:{limb_idx:02d},{JOINT_NAMES[idxA]}->{JOINT_NAMES[idxB]}")
        # 查找这两个点目前属于谁
        p_a = point_to_person.get(idA)
        p_b = point_to_person.get(idB)

        # 获取当前点的自身置信度（用于辅助决策）
        # all_peaks 结构：all_peaks[joint_idx] = [(x, y, score, global_id), ...]
        score_a = next((p[2] for p in all_peaks[idxA] if p[3] == idA), 0)
        score_b = next((p[2] for p in all_peaks[idxB] if p[3] == idB), 0)
        current_link_quality = paf_score + score_a + score_b

        # 情况 1：两个点都已经属于某个人了
        if p_a is not None and p_b is not None:
            if p_a == p_b: continue  # Already in same person

            # 【优化】：比较“合并后的预期质量”或“当前连线对双方的贡献”
            # 简单做法：谁拥有的该关节置信度低，或者谁当前总质量低，就合并到另一方
            # 但最稳妥的仍是：优先保留 PAF 积分高的那条骨架路径

            # 【修正】双向冲突检测：检查所有共同关节索引是否指向不同的点
            common = set(persons[p_a]['joints'].keys()) & set(persons[p_b]['joints'].keys())
            conflict = any(persons[p_a]['joints'][j] != persons[p_b]['joints'][j] for j in common)
            if not conflict:
                # 将小的/质量差的合并到大的/质量好的
                if persons[p_a]['score'] < persons[p_b]['score']:
                    p_a, p_b = p_b, p_a  # 确保 p_a 是较大的那个

                persons[p_a]['joints'].update(persons[p_b]['joints'])
                persons[p_a]['score'] += persons[p_b]['score'] + current_link_quality
                for pid in persons[p_b]['joints'].values(): point_to_person[pid] = p_a
                persons[p_b] = None
        # 情况 2：只有 idA 属于某人，尝试把 idB 加进去
        elif p_a is not None:
            if idxB not in persons[p_a]['joints']:
                persons[p_a]['joints'][idxB] = idB
                persons[p_a]['score'] += current_link_quality
                point_to_person[idB] = p_a
                # print(f"{idA}->{idB}，{idA}所属已有人物{p_a},添加{idB}")
        # 情况 3：只有 idB 属于某人，尝试把 idA 加进去
        elif p_b is not None:
            if idxA not in persons[p_b]['joints']:
                persons[p_b]['joints'][idxA] = idA
                persons[p_b]['score'] += current_link_quality
                point_to_person[idA] = p_b
                # print(f"{idA}->{idB}，{idB}所属已有人物{p_b},添加{idA}")
        # 情况 4：两个点都是新的，创建一个新的人
        else:
            # Create new person
            new_person = {'joints': {idxA: idA, idxB: idB}, 'score': current_link_quality}
            # 新元素的index索引位置
            person_index = len(persons)
            point_to_person[idA] = person_index
            point_to_person[idB] = person_index
            persons.append(new_person)
            # print(f"{idA}->{idB}所属新增人物{person_index}")

    # 清理被标记合并的旧对象，并过滤掉关键点太少的人
    persons = [p for p in persons if p is not None and len(p['joints']) >= MIN_KEYPOINTS]
    return persons


def persons_to_candidate_subset(persons, all_peaks):
    """Convert simplified objects back to OpenPose (candidate, subset) format."""
    candidate = []
    id_map = {}  # old_id -> new_candidate_idx
    for part_peaks in all_peaks:
        for p in part_peaks:
            id_map[p[3]] = len(candidate)
            candidate.append([p[0], p[1], p[2], p[3]])
    candidate = np.array(candidate)

    subset = []
    for p in persons:
        row = -1 * np.ones(NUM_JOINTS + 2)
        for j_idx, global_id in p['joints'].items():
            row[j_idx] = id_map.get(global_id, -1)
            row[-2] += candidate[int(row[j_idx]), 2] if row[j_idx] != -1 else 0
            row[-1] += 1
        subset.append(row)

    return candidate, np.array(subset) if subset else np.zeros((0, NUM_JOINTS + 2))


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
    """

    def __init__(self, model_path, width=32, input_size=256):
        self.input_size = input_size
        self.width = width
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build and load model (dual-branch: PAF + heatmap)
        self.model = HRNet(num_joints=NUM_JOINTS, num_limbs=NUM_LIMBS, width=width)
        state = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']

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

        img = cv2.resize(oriImg, (self.input_size, self.input_size)).astype(np.float32) / 255
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                paf_output, hm_output = self.model(img)
        paf_np = paf_output.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.float32)  # (H, W, C)
        heatmap_np = hm_output.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.float32)

        # 【可选】在64x64上做粗略阈值过滤
        heatmap_np[heatmap_np < HEATMAP_FILTER] = 0  # 提前过滤噪声

        # Resize to scaled image size, then to original size
        paf_output = cv2.resize(paf_np, (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap_output = cv2.resize(heatmap_np, (w, h), interpolation=cv2.INTER_CUBIC)

        # Save per-joint heatmaps
        for part in range(NUM_JOINTS):
            hm_single = heatmap_output[:, :, part].astype(np.float32)
            hm_norm = cv2.normalize(hm_single, None, 0, 255, cv2.NORM_MINMAX)
            hm_colored = cv2.applyColorMap(hm_norm.astype(np.uint8), cv2.COLORMAP_JET)
            blend = cv2.addWeighted(oriImg, 0.6, hm_colored, 0.4, 0)
            cv2.imwrite(os.path.join('../output/hrnet_body_pose', f'heatmap_joint_{part:02d}.jpg'), blend)

        # Step 1: Peak Detection with Distance Merging
        all_peaks = []
        peak_counter = 0
        for part in range(NUM_JOINTS):
            map_ori = heatmap_output[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=GAUSSIAN_FILTER_SIGMA)

            # 替代手动的 4-neighbor 比较
            neighborhood = np.ones((7, 7))  # 7x7 窗口 NMS
            local_max = maximum_filter(one_heatmap, footprint=neighborhood) == one_heatmap
            mask = local_max & (one_heatmap > PEAK_THRESHOLD)

            # 查找每个维度上非零（True）元素的索引（y,x）坐标，通过zip转换为（x,y）
            peaks = list(zip(np.nonzero(mask)[1], np.nonzero(mask)[0]))
            #  map_ori 是NumPy数组，需要用 [行, 列] 即，(y,x)，将score加入到peaks，形成（x,y,score）
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]

            # Distance-based deduplication
            valid_peaks = []
            for p in peaks_with_score:
                is_dup = False
                for i, kept in enumerate(valid_peaks):
                    if math.hypot(p[0] - kept[0], p[1] - kept[1]) < MIN_DIST_THRESHOLD:
                        if p[2] > kept[2]: valid_peaks[i] = p
                        is_dup = True;
                        break
                if not is_dup: valid_peaks.append(p)

            # 为当前关节的点分配 ID 并加入 all_peaks
            current_part_peaks = []
            for p in valid_peaks:
                x, y, score = p
                current_part_peaks.append((x, y, score, peak_counter))
                peak_counter += 1
            all_peaks.append(current_part_peaks)  # all_peaks[part] 存储该关节的所有点

        # Step 2 & 3: PAF Scoring & Greedy Assembly
        links = group_keypoints_by_paf(all_peaks, paf_output, (h, w))
        persons = assemble_persons_simple(all_peaks, links)

        return persons_to_candidate_subset(persons, all_peaks)
