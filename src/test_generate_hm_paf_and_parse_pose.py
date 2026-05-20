"""Pre-compute ground truth heatmaps, PAFs, and masks for the COCO dataset.

Saves each image's ground truth as a .npz file for faster training.
Run once before training: python preprocess_dataset.py
"""

import os
import numpy as np
from tqdm import tqdm
import cv2

from src import util
from utils.HRNetCocoDataset import HRNetCocoDataset
from config import NUM_JOINTS, NUM_LIMBS, SKELETONS
from scipy.ndimage import gaussian_filter
from hrnet_body_pose import assemble_persons,group_keypoints_by_paf

PEAK_THRESHOLD = 0.1  # Heatmap peak detection threshold


def deal(heatmap, paf, ori_height, ori_width):
    # ── Step 2: Peak detection ──
    # Same logic as body.py: Gaussian smooth + NMS
    all_peaks = []
    peak_counter = 0

    for part in range(NUM_JOINTS):
        map_ori = heatmap[:, :, part]
        one_heatmap = gaussian_filter(map_ori, sigma=1)

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
        all_peaks, paf, (ori_height, ori_width)
    )

    # ── Step 4: Person assembly ──
    candidate, subset = assemble_persons(all_peaks, connection_all)

    return candidate, subset


def generate_hm_paf_and_parse_pose(data_dir="../data", split="val2017",
                                   output_dir="../output/test_generate_and_parse"):
    os.makedirs(output_dir, exist_ok=True)

    dataset = HRNetCocoDataset(
        data_dir,
        split,
        input_size=256,
        heatmap_size=64,
        sigma=1.0,
        paf_sigma=2.0,
        augment=False,
        filter_key_points_nums=10
    )
    output_split_dir = os.path.join(output_dir, split)
    os.makedirs(output_split_dir, exist_ok=True)

    print(f"Testing {len(dataset)} images...")

    for idx in tqdm(range(len(dataset))):
        img_tensor, paf_t, hm_t, mask_t = dataset[idx]
        img = img_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
        img = cv2.resize(img, (1080, 1080),
                         interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.uint8)

        paf_np = paf_t.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.float32)  # (H, W, C)
        heatmap_np = hm_t.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.float32) # (H, W, C)

        paf_np = cv2.resize(paf_np, img.shape[0:2], interpolation=cv2.INTER_CUBIC)
        heatmap_np = cv2.resize(heatmap_np, img.shape[0:2], interpolation=cv2.INTER_CUBIC)

        candidate, subset = deal(heatmap_np, paf_np, img.shape[0], img.shape[1])
        canvas = util.draw_bodypose(img, candidate, subset)

        # Save result
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, os.path.basename(str(idx)+".jpg"))
        cv2.imwrite(save_path, canvas)
        print(f"Result saved to {save_path}")

        # Display
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 2, 1)
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.title('Original')
        # plt.axis('off')
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        # plt.title(f'HRNet Multi-Person Pose ({len(subset)} persons)')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()

        # Print results
        print(f"Detected {len(subset)} persons, {len(candidate)} total keypoints")
        for i in range(len(subset)):
            n_kpts = int(subset[i][-1])
            avg_conf = subset[i][-2] / subset[i][-1] if subset[i][-1] > 0 else 0
            print(f"  Person {i}: {n_kpts} keypoints, avg confidence={avg_conf:.3f}")

    print(f"Preprocessed data saved to {output_split_dir}")


if __name__ == "__main__":
    generate_hm_paf_and_parse_pose()
