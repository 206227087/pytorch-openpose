"""Single-image body pose estimation demo."""

import cv2
import matplotlib.pyplot as plt

from src import util
from src.body import Body

body_estimation = Body('model/body_pose_model.pth')
# hand_estimation = Hand('model/hand_pose_model.pth')

test_image = 'images/demo.jpg'
oriImg = cv2.imread(test_image)
candidate, subset = body_estimation(oriImg)
canvas = util.draw_bodypose(oriImg, candidate, subset)

# Hand detection (uncomment to enable)
# hands_list = util.handDetect(candidate, subset, oriImg)
# all_hand_peaks = []
# for x, y, w, is_left in hands_list:
#     peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
#     peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
#     peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
#     all_hand_peaks.append(peaks)
all_hand_peaks = []

canvas = util.draw_handpose(canvas, all_hand_peaks)
plt.imshow(canvas[:, :, [2, 1, 0]])
plt.axis('off')
plt.show()
