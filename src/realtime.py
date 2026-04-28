"""Real-time pose estimation system for video and camera input.

Provides a unified RealTimePoseEstimator class supporting both OpenPose
(VGG backbone) and HRNet backbones, with optional hand pose estimation.
"""

import cv2
import numpy as np
import torch

from src.body import Body
from src.hand import Hand
from src import util


# OpenPose 18-joint skeleton connections for visualization
BODY_SKELETON = [
    (1, 2), (1, 5),   # neck to shoulders
    (2, 3), (3, 4),   # right arm
    (5, 6), (6, 7),   # left arm
    (1, 8), (1, 11),  # neck to hips
    (8, 9), (9, 10),  # right leg
    (11, 12), (12, 13),  # left leg
    (1, 0),            # neck to nose
    (0, 14), (14, 16),  # right eye-ear
    (0, 15), (15, 17),  # left eye-ear
    (2, 16), (5, 17),   # shoulders to ears
]


class RealTimePoseEstimator:
    """Real-time multi-person pose estimation for video streams.

    Supports both OpenPose (body + hand) and HRNet backbones with
    a clean interface for processing video files or camera feeds.

    Args:
        body_model_path: path to body pose model weights.
        hand_model_path: path to hand pose model weights (None to disable).
        backend: 'openpose' or 'hrnet' (default 'openpose').
        input_size: model input size (default 368 for OpenPose, 256 for HRNet).
        device: compute device (default 'cuda' if available).
    """

    def __init__(self, body_model_path, hand_model_path=None,
                 backend='openpose', input_size=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.backend = backend
        self.input_size = input_size or (368 if backend == 'openpose' else 256)

        # Load body model
        if backend == 'openpose':
            self.body = Body(body_model_path)
        elif backend == 'hrnet':
            from src.model import HRNet
            self.body_model = HRNet(num_joints=17, width=32)
            state = torch.load(body_model_path, map_location=self.device)
            self.body_model.load_state_dict(state)
            self.body_model.to(self.device).eval()
            self.body = None  # HRNet uses direct model
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Load hand model
        self.hand = None
        if hand_model_path:
            self.hand = Hand(hand_model_path)

    def process_frame(self, frame):
        """Process a single video frame.

        Args:
            frame: BGR image (numpy array from cv2).

        Returns:
            Annotated frame with pose skeletons drawn.
        """
        if self.backend == 'openpose':
            return self._process_openpose(frame)
        else:
            return self._process_hrnet(frame)

    def _process_openpose(self, frame):
        """Process frame with OpenPose backend (body + optional hand)."""
        candidate, subset = self.body(frame)
        canvas = util.draw_bodypose(frame, candidate, subset)

        if self.hand is not None:
            hands_list = util.handDetect(candidate, subset, frame)
            for x, y, w, is_left in hands_list:
                hand_crop = frame[y:y+w, x:x+w]
                if hand_crop.size == 0:
                    continue
                hand_keypoints = self.hand(hand_crop)
                canvas = util.draw_handpose(canvas, hand_keypoints)

        return canvas

    def _process_hrnet(self, frame):
        """Process frame with HRNet backend (body only)."""
        # Preprocess
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img_tensor = torch.from_numpy(
            img.transpose(2, 0, 1).astype(np.float32) / 255.0
        ).unsqueeze(0).to(self.device)

        # Inference
        with torch.inference_mode():
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                output = self.body_model(img_tensor)

        # Postprocess: extract keypoints from heatmaps
        keypoints = self._extract_keypoints(output, frame.shape[:2])
        canvas = self._draw_hrnet_pose(frame, keypoints)
        return canvas

    def _extract_keypoints(self, output, original_shape):
        """Extract keypoint coordinates from HRNet heatmap output.

        Args:
            output: (1, K, H, W) heatmap tensor.
            original_shape: (height, width) of original frame.

        Returns:
            List of (x, y) keypoint coordinates.
        """
        heatmaps = output.squeeze(0).cpu().numpy()
        K, H, W = heatmaps.shape
        scale_y = original_shape[0] / H
        scale_x = original_shape[1] / W

        keypoints = []
        for k in range(K):
            y, x = np.unravel_index(heatmaps[k].argmax(), (H, W))
            keypoints.append((int(x * scale_x), int(y * scale_y)))
        return keypoints

    def _draw_hrnet_pose(self, frame, keypoints):
        """Draw HRNet pose skeleton on frame.

        Args:
            frame: BGR image.
            keypoints: list of (x, y) coordinates.

        Returns:
            Annotated frame.
        """
        canvas = frame.copy()
        for i, j in BODY_SKELETON:
            if i < len(keypoints) and j < len(keypoints):
                p1, p2 = keypoints[i], keypoints[j]
                if p1[0] > 0 and p2[0] > 0:
                    cv2.line(canvas, p1, p2, (0, 255, 0), 2)
        for x, y in keypoints:
            if x > 0 and y > 0:
                cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)
        return canvas

    def run(self, source=0, output_path=None, display=True):
        """Run real-time pose estimation on a video source.

        Args:
            source: video file path or camera index (default 0 for webcam).
            output_path: optional path to save annotated video.
            display: whether to display frames (default True).

        Returns:
            None. Press 'q' to quit.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        writer = None
        if output_path:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.process_frame(frame)

                if writer is not None:
                    writer.write(result)

                if display:
                    cv2.imshow('Pose Estimation', result)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()
