import cv2
import copy
import numpy as np

from src import util
from src.body import Body

# 1. 初始化模型
body_estimation = Body('./model/body_pose_model.pth')

# 2. 读取图片
test_image = 'images/demo.jpg'
oriImg = cv2.imread(test_image)

if oriImg is None:
    print(f"Error: Could not load image from {test_image}")
else:
    print("Running pose estimation...")
    # 3. 推理
    candidate, subset = body_estimation(oriImg)
    
    print(f"Candidate shape: {candidate.shape}")
    print(f"Subset shape: {subset.shape}")

    # 4. 绘图
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    # 5. 保存结果 (这样就不用依赖 plt 弹窗了)
    output_path = 'result_debug.jpg'
    cv2.imwrite(output_path, canvas)
    print(f"Success! Result saved to {output_path}")
    
    # 如果你还是想弹窗看
    # cv2.imshow('Result', canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()