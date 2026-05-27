"""Shared configuration constants for OpenPose body pose estimation.

Used by both training (train.py) and inference (src/body.py).
"""

# ─── Joint Definitions ────────────────────────────────────────────────────────
# 18 key points (COCO 17 key points + 1 neck point):
NUM_JOINTS = 18

# Keypoint names for labeling
JOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'neck'
]

# key points 索引映射
# 0:nose 1:left_eye 2:right_eye 3:left_ear 4:right_ear
# 5:left_shoulder 6:right_shoulder 7:left_elbow 8:right_elbow
# 9:left_wrist 10:right_wrist 11:left_hip 12:right_hip
# 13:left_knee 14:right_knee 15:left_ankle 16:right_ankle
# 17:neck (should calculate mid point from left_shoulder and right_shoulder)
KEYPOINT_MAP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

# COCO key points 左右对称映射（翻转时需要交换）
# 0:nose(对称) 1:left_eye<->2:right_eye 3:left_ear<->4:right_ear
# 5:left_shoulder<->6:right_shoulder 7:left_elbow<->8:right_elbow
# 9:left_wrist<->10:right_wrist 11:left_hip<->12:right_hip
# 13:left_knee<->14:right_knee 15:left_ankle<->16:right_ankle
# 17:neck (should calculate mid point from left_shoulder and right_shoulder)
KEYPOINT_FLIP_MAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17]

# COCO skeleton connections (18 limbs)
SKELETONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # face
    (0, 17), (17, 5), (17, 6),  # neck
    (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12),  # torso
    (11, 12), (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]
NUM_LIMBS = len(SKELETONS)  # 18
NUM_PAF_CHANNELS = NUM_LIMBS * 2  # 36

# ─── Image / Preprocessing Parameters ─────────────────────────────────────────
INPUT_SIZE = 368
HEATMAP_SIZE = INPUT_SIZE // 8  # 46
STRIDE = 8
PAD_VALUE = 128

# ─── Ground Truth Generation Parameters ───────────────────────────────────────
SIGMA = 3.0  # Gaussian spread for heatmaps (in heatmap-space pixels)
PAF_SIGMA = 1.0  # PAF limb width (in heatmap-space pixels)

# ─── Inference Parameters ─────────────────────────────────────────────────────
SCALE_SEARCH = [0.5, 1.0, 1.5, 2.0]
PEAK_THRESHOLD = 0.15  # heatmap peak detection threshold
PAF_SCORE_THRESHOLD = 0.05  # minimum average PAF score for a valid connection
MID_NUM = 10  # number of sample points along each PAF for scoring

# ─── Model Output Channels ────────────────────────────────────────────────────
NUM_HEATMAP_CHANNELS = NUM_JOINTS  # (18 joints)

# ─── DEBUG Config ─────────────────────────────────────────────────────────────
DEBUG = True
DEBUG_COMPARE_GT_DIR = 'output/compare_gt'  # GT visualization output directory
