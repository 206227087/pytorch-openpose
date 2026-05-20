"""Utility functions for OpenPose visualization and hand detection.
"""

import math

import cv2
import numpy as np

from config import NUM_JOINTS,SKELETONS
# 定义二十种颜色


def draw_bodypose(canvas, candidate, subset):
    """Draw body keypoints and limb connections on an image.

    Args:
        canvas: Image to draw on (modified in-place)
        candidate: (N, 4) array of [x, y, score, id]
        subset: (M, 20) array of person data

    Returns:
        canvas: Image with drawn poses
    """
    canvas = canvas.copy()
    for person in subset:
        # 每个人一个随机颜色
        person_color = tuple(int(c) for c in np.random.randint(0, 255, size=3))
        # Draw limbs
        for limb in SKELETONS:
            idx_a, idx_b = limb
            joint_a = int(person[idx_a])
            joint_b = int(person[idx_b])
            if joint_a == -1 or joint_b == -1:
                continue
            X = candidate[[joint_a, joint_b]][:, 0]
            Y = candidate[[joint_a, joint_b]][:, 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            stickwidth = 2
            polygon = cv2.ellipse2Poly(
                (int(mX), int(mY)), (int(length / 2), stickwidth),
                int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, person_color)
        # Draw joints
        for idx in range(NUM_JOINTS):
            joint = int(person[idx])
            if joint == -1:
                continue
            x, y = candidate[joint][:2]
            cv2.circle(canvas, (int(x), int(y)), 3, person_color, thickness=2)
            cv2.putText(canvas, str(idx), (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, person_color, 2)
    return canvas
