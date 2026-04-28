"""HRNet-based body pose estimation inference pipeline.

Provides a BodyHRNet class that uses the HRNet backbone for single-person
keypoint detection, as an alternative to the OpenPose VGG-based model.
"""

import cv2
import numpy as np
import torch

from src.model import HRNet
from src.inference import MixedPrecisionInference


class BodyHRNet:
    """HRNet body pose estimator.

    Uses the High-Resolution Network backbone for 17-keypoint COCO
    pose estimation. Supports mixed-precision inference for speed.

    Args:
        model_path: path to HRNet weights file.
        num_joints: number of keypoints (default 17 for COCO).
        width: HRNet width (32 for W32, 48 for W48).
        input_size: input image size (default 256).
        use_amp: enable automatic mixed precision (default True).
    """

    # COCO 17-keypoint names
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    ]

    # COCO skeleton connections
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),        # face
        (5, 6), (5, 7), (7, 9), (6, 8),        # arms
        (8, 10), (5, 11), (6, 12),             # torso
        (11, 12), (11, 13), (13, 15),          # left leg
        (12, 14), (14, 16),                     # right leg
    ]

    def __init__(self, model_path, num_joints=17, width=32,
                 input_size=256, use_amp=True):
        self.num_joints = num_joints
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build and load model
        self.model = HRNet(num_joints=num_joints, width=width)
        state = torch.load(model_path, map_location=self.device)

        # Handle different state dict formats
        if 'state_dict' in state:
            state = state['state_dict']
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        # Optionally wrap with AMP
        if use_amp and torch.cuda.is_available():
            self._infer = MixedPrecisionInference(self.model, self.device)
        else:
            self._infer = self.model

    def __call__(self, oriImg):
        """Run HRNet body pose estimation on an image.

        Args:
            oriImg: BGR image (numpy array from cv2).

        Returns:
            keypoints: (K, 3) array of [x, y, score] for each keypoint.
            heatmap: (K, H, W) raw heatmap output.
        """
        h, w = oriImg.shape[:2]

        # Preprocess
        img = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img_tensor = torch.from_numpy(
            img.transpose(2, 0, 1).astype(np.float32) / 255.0
        ).unsqueeze(0).to(self.device)

        # Normalize (ImageNet mean/std)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # Inference
        with torch.inference_mode():
            output = self._infer(img_tensor)

        # Postprocess
        heatmap = output.squeeze(0).cpu().numpy()  # (K, H, W)
        keypoints = self._decode_heatmap(heatmap, (h, w))

        return keypoints, heatmap

    def _decode_heatmap(self, heatmap, original_size):
        """Decode heatmap to keypoint coordinates.

        Uses argmax for initial location, then sub-pixel refinement
        via quarter-offset around the peak.

        Args:
            heatmap: (K, H, W) numpy array.
            original_size: (height, width) of original image.

        Returns:
            keypoints: (K, 3) array of [x, y, score].
        """
        K, H, W = heatmap.shape
        scale_y = original_size[0] / H
        scale_x = original_size[1] / W

        keypoints = np.zeros((K, 3))
        for k in range(K):
            hm = heatmap[k]
            score = hm.max()

            # Argmax location
            y, x = np.unravel_index(hm.argmax(), (H, W))

            # Sub-pixel refinement using neighboring values
            if 0 < x < W - 1 and 0 < y < H - 1:
                dx = (hm[y, x + 1] - hm[y, x - 1]) / 2.0
                dy = (hm[y + 1, x] - hm[y - 1, x]) / 2.0
                dxx = hm[y, x + 1] + hm[y, x - 1] - 2 * hm[y, x]
                dyy = hm[y + 1, x] + hm[y - 1, x] - 2 * hm[y, x]
                if abs(dxx) > 1e-6:
                    x -= dx / dxx
                if abs(dyy) > 1e-6:
                    y -= dy / dyy

            keypoints[k, 0] = x * scale_x
            keypoints[k, 1] = y * scale_y
            keypoints[k, 2] = score

        return keypoints

    def draw_pose(self, image, keypoints):
        """Draw HRNet keypoints and skeleton on image.

        Args:
            image: BGR image.
            keypoints: (K, 3) array of [x, y, score].

        Returns:
            Annotated image.
        """
        canvas = image.copy()

        # Draw skeleton
        for i, j in self.SKELETON:
            if keypoints[i, 2] > 0.1 and keypoints[j, 2] > 0.1:
                p1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
                p2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
                cv2.line(canvas, p1, p2, (0, 255, 0), 2)

        # Draw keypoints
        for k in range(len(keypoints)):
            if keypoints[k, 2] > 0.1:
                x, y = int(keypoints[k, 0]), int(keypoints[k, 1])
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), -1)

        return canvas
