"""Shared configuration constants for OpenPose body pose estimation.

Used by both training (train.py) and inference (src/body.py).
"""

# ─── Joint Definitions ────────────────────────────────────────────────────────
# OpenPose 18 joints:
#   0: nose,  1: neck,  2: r_shoulder,  3: r_elbow,  4: r_wrist,
#   5: l_shoulder,  6: l_elbow,  7: l_wrist,  8: r_hip,  9: r_knee,
#  10: r_ankle,  11: l_hip,  12: l_knee,  13: l_ankle,
#  14: r_eye,  15: l_eye,  16: r_ear,  17: l_ear
NUM_JOINTS = 18

# COCO keypoint order -> OpenPose order mapping
# COCO: nose(0) l_eye(1) r_eye(2) l_ear(3) r_ear(4)
#       l_shoulder(5) r_shoulder(6) l_elbow(7) r_elbow(8)
#       l_wrist(9) r_wrist(10) l_hip(11) r_hip(12)
#       l_knee(13) r_knee(14) l_ankle(15) r_ankle(16)
# -1 means the joint doesn't exist in COCO (neck is computed as midpoint of shoulders)
COCO_TO_OPENPOSE = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

# ─── Limb (PAF) Definitions ───────────────────────────────────────────────────
# PAF limb connections (pairs of OpenPose joint indices)
# 19 limbs total for COCO 18 joints (including neck)
LIMBS = [
    (1, 2), (1, 5),  # neck to shoulders
    (2, 3), (3, 4),  # right arm
    (5, 6), (6, 7),  # left arm
    (1, 8), (1, 11),  # neck to hips
    (8, 9), (9, 10),  # right leg
    (11, 12), (12, 13),  # left leg
    (1, 0),  # neck to nose
    (0, 14), (14, 16),  # right eye-ear
    (0, 15), (15, 17),  # left eye-ear
    (2, 16), (5, 17),  # shoulders to ears
]

NUM_LIMBS = len(LIMBS)  # 19
NUM_PAF_CHANNELS = NUM_LIMBS * 2  # 38 (x, y per limb)

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
PEAK_THRESHOLD = 0.1  # heatmap peak detection threshold
PAF_SCORE_THRESHOLD = 0.05  # minimum average PAF score for a valid connection
MID_NUM = 10  # number of sample points along each PAF for scoring

# ─── Model Output Channels ────────────────────────────────────────────────────
NUM_HEATMAP_CHANNELS = NUM_JOINTS + 1  # 19 (18 joints + background)

# ─── DEBUG Config ─────────────────────────────────────────────────────────────
DEBUG = True
DEBUG_COMPARE_GT_DIR = 'output/compare_gt'  # GT visualization output directory
