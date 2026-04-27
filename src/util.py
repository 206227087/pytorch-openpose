"""Utility functions for OpenPose visualization and hand detection.
"""

import math

import cv2
import numpy as np
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def padRightDownCorner(img, stride, padValue):
    """Pad image on right and bottom to make dimensions divisible by stride.

    Deprecated: use src.preprocessing.pad_image instead.
    Kept for backward compatibility.
    """
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = (stride - (h % stride)) if (h % stride != 0) else 0  # down
    pad[3] = (stride - (w % stride)) if (w % stride != 0) else 0  # right

    img_padded = img.copy()
    img_padded = cv2.copyMakeBorder(img_padded, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=[padValue, padValue, padValue])

    return img_padded, pad


def transfer(model, model_weights):
    """Transfer Caffe model weights to PyTorch by stripping key prefixes.

    Caffe-converted weights have an extra prefix segment in each key name.
    This function strips the first segment (e.g., 'model.' -> '').
    """
    model_dict = model.state_dict()
    pretrained_dict = {}
    for k, v in model_weights.items():
        # Strip the first segment of the key
        key = k[k.find('.') + 1:]
        if key in model_dict:
            pretrained_dict[key] = v
    model_dict.update(pretrained_dict)
    return model_dict


# --- Visualization -----------------------------------------------------------

# Body joint colors (BGR format for OpenCV)
JOINT_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
    [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
    [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
    [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
    [255, 0, 170], [255, 0, 85],
]

# Body limb connections for drawing (pairs of joint indices)
LIMB_PAIRS = [
    [0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8],
    [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [0, 14], [0, 15],
    [14, 16], [15, 17],
]

# Hand finger edges for drawing
HAND_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15],
    [15, 16], [0, 17], [17, 18], [18, 19], [19, 20],
]


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
        # Draw limbs
        for limb in LIMB_PAIRS:
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
            stickwidth = 4
            polygon = cv2.ellipse2Poly(
                (int(mX), int(mY)), (int(length / 2), stickwidth),
                int(angle), 0, 360, 1)
            color = JOINT_COLORS[limb[0]]
            cv2.fillConvexPoly(canvas, polygon, color)
        # Draw joints
        for idx in range(18):
            joint = int(person[idx])
            if joint == -1:
                continue
            x, y = candidate[joint][:2]
            cv2.circle(canvas, (int(x), int(y)), 4, JOINT_COLORS[idx], thickness=-1)
    return canvas


def draw_handpose(canvas, all_hand_peaks, show_number=False):
    """Draw hand keypoints and connections using matplotlib.

    Args:
        canvas: Image to draw on
        all_hand_peaks: List of (21, 2) arrays of hand keypoint coordinates
        show_number: Whether to show keypoint numbers

    Returns:
        canvas: Image with drawn hand poses
    """
    if not all_hand_peaks:
        return canvas

    fig = Figure(figsize=(canvas.shape[1] / 100.0, canvas.shape[0] / 100.0), dpi=100)
    ax = fig.add_subplot(111)
    ax.imshow(canvas[:, :, [2, 1, 0]] / 255.0)

    for peaks in all_hand_peaks:
        for edge in HAND_EDGES:
            ax.plot([peaks[edge[0]][0], peaks[edge[1]][0]],
                    [peaks[edge[0]][1], peaks[edge[1]][1]],
                    color='w', linewidth=2)
        for i, peak in enumerate(peaks):
            ax.plot(peak[0], peak[1], 'r.', markersize=8)
            if show_number:
                ax.text(peak[0], peak[1], str(i), color='blue', fontsize=8)

    ax.axis('off')
    fig.tight_layout()
    canvas = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    canvas = canvas.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return canvas


def draw_handpose_by_opencv(canvas, peaks, show_number=False):
    """Draw hand keypoints and connections using OpenCV directly.

    Args:
        canvas: Image to draw on (modified in-place)
        peaks: (21, 2) array of hand keypoint coordinates
        show_number: Whether to show keypoint numbers

    Returns:
        canvas: Image with drawn hand poses
    """
    for edge in HAND_EDGES:
        cv2.line(canvas, tuple(peaks[edge[0]].astype(int)),
                 tuple(peaks[edge[1]].astype(int)), (255, 255, 255), 2)
    for i, peak in enumerate(peaks):
        cv2.circle(canvas, tuple(peak.astype(int)), 4, (0, 0, 255), -1)
        if show_number:
            cv2.putText(canvas, str(i), tuple(peak.astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return canvas


# --- Hand Detection -----------------------------------------------------------

def handDetect(candidate, subset, oriImg):
    """Detect hand bounding boxes from body pose keypoints.

    Uses wrist, elbow, and shoulder positions to estimate hand location
    and size. Based on the OpenPose hand detector implementation.

    Args:
        candidate: (N, 4) array of body keypoint [x, y, score, id]
        subset: (M, 20) array of person data
        oriImg: Original image (for bounds checking)

    Returns:
        List of [x, y, w, is_left] for each detected hand,
        where (x, y) is top-left corner and w is the square box size.
    """
    ratioWristElbow = 0.33
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]

    for person in subset.astype(int):
        has_left = np.sum(person[[5, 6, 7]] == -1) == 0
        has_right = np.sum(person[[2, 3, 4]] == -1) == 0
        if not (has_left or has_right):
            continue

        hands = []
        if has_left:
            left_shoulder_idx, left_elbow_idx, left_wrist_idx = person[[5, 6, 7]]
            x1, y1 = candidate[left_shoulder_idx][:2]
            x2, y2 = candidate[left_elbow_idx][:2]
            x3, y3 = candidate[left_wrist_idx][:2]
            hands.append([x1, y1, x2, y2, x3, y3, True])
        if has_right:
            right_shoulder_idx, right_elbow_idx, right_wrist_idx = person[[2, 3, 4]]
            x1, y1 = candidate[right_shoulder_idx][:2]
            x2, y2 = candidate[right_elbow_idx][:2]
            x3, y3 = candidate[right_wrist_idx][:2]
            hands.append([x1, y1, x2, y2, x3, y3, False])

        for x1, y1, x2, y2, x3, y3, is_left in hands:
            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            x -= width / 2
            y -= width / 2
            if x < 0: x = 0
            if y < 0: y = 0
            width1 = width
            width2 = width
            if x + width > image_width: width1 = image_width - x
            if y + width > image_height: width2 = image_height - y
            width = min(width1, width2)
            if width >= 20:
                detect_result.append([int(x), int(y), int(width), is_left])

    return detect_result


def npmax(array):
    """Get (row, col) index of the maximum value in a 2D array."""
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j
