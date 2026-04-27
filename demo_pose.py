"""Single-image body pose estimation demo."""

import cv2
import matplotlib.pyplot as plt

from src import util
from src.body import Body

if __name__ == '__main__':
    body_estimation = Body('model/epoch0004_loss0.1210.pth')

    test_image = 'images/ski.jpg'
    oriImg = cv2.imread(test_image)
    candidate, subset = body_estimation(oriImg)
    canvas = util.draw_bodypose(oriImg, candidate, subset)

    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.show()
