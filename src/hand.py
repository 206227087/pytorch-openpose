"""Hand keypoint estimation inference pipeline.

Detects 21 hand keypoints from a cropped hand image.
"""

import cv2
import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter
from skimage.measure import label

from src.config import INPUT_SIZE, STRIDE, SCALE_SEARCH
from src.model import handpose_model
from src import preprocessing
from src import util

NUM_HAND_KEYPOINTS = 21
HAND_HEATMAP_CHANNELS = 22  # 21 keypoints + background
HAND_PEAK_THRESHOLD = 0.05


class Hand:
    """Hand keypoint estimator.

    Usage:
        hand = Hand('model/hand_pose_model.pth')
        peaks = hand(hand_crop_image)
    """

    def __init__(self, model_path):
        self.model = handpose_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        model_dict = util.transfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def __call__(self, oriImg):
        """Run hand keypoint estimation on a cropped hand image.

        Args:
            oriImg: Cropped hand image (H, W, 3) in BGR order

        Returns:
            peaks: (21, 2) array of (x, y) coordinates for each keypoint
        """
        multiplier = [s * INPUT_SIZE / oriImg.shape[0] for s in SCALE_SEARCH]
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], HAND_HEATMAP_CHANNELS))

        for scale in multiplier:
            data, padded_shape, pad = preprocessing.prepare_model_input(
                oriImg, scale, STRIDE)

            with torch.no_grad():
                output = self.model(data).cpu().numpy()

            heatmap = np.transpose(np.squeeze(output), (1, 2, 0))
            heatmap = preprocessing.resize_output(
                heatmap, STRIDE, padded_shape, pad, oriImg.shape)
            heatmap_avg += heatmap / len(multiplier)

        # Peak detection with connected component filtering
        all_peaks = []
        for part in range(NUM_HAND_KEYPOINTS):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=3)
            binary = np.ascontiguousarray(one_heatmap > HAND_PEAK_THRESHOLD, dtype=np.uint8)

            if np.sum(binary) == 0:
                all_peaks.append([0, 0])
                continue

            # Find the strongest connected component
            label_img, label_numbers = label(binary, return_num=True, connectivity=binary.ndim)
            max_index = np.argmax([
                np.sum(map_ori[label_img == i])
                for i in range(1, label_numbers + 1)
            ]) + 1
            label_img[label_img != max_index] = 0
            map_ori[label_img == 0] = 0

            y, x = util.npmax(map_ori)
            all_peaks.append([x, y])

        return np.array(all_peaks)
