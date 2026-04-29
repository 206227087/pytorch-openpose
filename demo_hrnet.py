"""HRNet multi-person body pose estimation demo.

Demonstrates single-image and real-time multi-person pose estimation using
the HRNet backbone with PAF+heatmap dual-branch output. Also shows model
profiling and comparison with OpenPose.

Usage:
  python demo_hrnet.py image --image images/demo.jpg --model model/hrnet_w32.pth
  python demo_hrnet.py profile
  python demo_hrnet.py realtime --model model/hrnet_w32.pth
"""

import argparse

import cv2
import matplotlib.pyplot as plt
import torch

from src.hrnet_body_pose import BodyHRNetPose, convert_to_openpose_format
from src.inference import profile_model
from src.model import HRNet


def demo_image(image_path, model_path, width=32, input_size=256):
    """Run HRNet multi-person pose estimation on a single image."""
    body = BodyHRNetPose(model_path, width=width, input_size=input_size)

    oriImg = cv2.imread(image_path)
    if oriImg is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    candidate, subset = body(oriImg)

    # Convert to OpenPose format for drawing
    candidate_op, subset_op = convert_to_openpose_format(candidate.copy(), subset)

    # Draw using util.draw_bodypose
    from src import util
    canvas = util.draw_bodypose(oriImg, candidate_op, subset_op)

    # Display
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title(f'HRNet Multi-Person Pose ({len(subset)} persons)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Print results
    print(f"Detected {len(subset)} persons, {len(candidate)} total keypoints")
    for i in range(len(subset)):
        n_kpts = int(subset[i][-1])
        avg_conf = subset[i][-2] / subset[i][-1] if subset[i][-1] > 0 else 0
        print(f"  Person {i}: {n_kpts} keypoints, avg confidence={avg_conf:.3f}")


def demo_profile(width=32, input_size=256):
    """Profile HRNet model speed and compare with OpenPose."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Profile HRNet-W32
    print(f"--- HRNet-W{width} Profiling ---")
    hrnet = HRNet(num_joints=17, num_limbs=16, width=width).to(device).eval()
    stats = profile_model(hrnet, input_shape=(1, 3, input_size, input_size),
                          device=device)
    print(f"  Parameters: {stats['params_M']:.2f}M")
    print(f"  Avg inference: {stats['avg_ms']:.1f} ms")
    print(f"  FPS: {stats['fps']:.1f}")
    if stats['gpu_mem_MB'] > 0:
        print(f"  GPU memory: {stats['gpu_mem_MB']:.0f} MB")

    # Profile OpenPose for comparison
    print(f"\n--- OpenPose Profiling ---")
    from src.model import bodypose_model
    openpose = bodypose_model().to(device).eval()
    stats_op = profile_model(openpose, input_shape=(1, 3, 368, 368),
                             device=device)
    print(f"  Parameters: {stats_op['params_M']:.2f}M")
    print(f"  Avg inference: {stats_op['avg_ms']:.1f} ms")
    print(f"  FPS: {stats_op['fps']:.1f}")
    if stats_op['gpu_mem_MB'] > 0:
        print(f"  GPU memory: {stats_op['gpu_mem_MB']:.0f} MB")

    # Comparison
    print(f"\n--- Comparison ---")
    speedup = stats_op['avg_ms'] / stats['avg_ms']
    param_ratio = stats_op['params_M'] / stats['params_M']
    print(f"  HRNet is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than OpenPose")
    print(f"  HRNet has {param_ratio:.1f}x the parameters of OpenPose")


def demo_realtime(model_path, source=0, width=32, input_size=256):
    """Run real-time HRNet multi-person pose estimation on video/camera."""
    body = BodyHRNetPose(model_path, width=width, input_size=input_size)
    from src import util

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    print("Press 'q' to quit")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        candidate, subset = body(frame)
        candidate_op, subset_op = convert_to_openpose_format(candidate.copy(), subset)
        canvas = util.draw_bodypose(frame, candidate_op, subset_op)

        cv2.imshow('HRNet Multi-Person Pose', canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HRNet multi-person pose estimation demo")
    sub = parser.add_subparsers(dest='command')

    # Image demo
    img_parser = sub.add_parser('image', help='Run on a single image')
    img_parser.add_argument('--image', default='images/demo.jpg', help='Input image path')
    img_parser.add_argument('--model', default='checkpoints/hrnet_w48_epoch0005_loss0.1607.pth', help='HRNet model weights')
    img_parser.add_argument('--width', type=int, default=48, help='HRNet width (32 or 48)')
    img_parser.add_argument('--input_size', type=int, default=256)

    # Profile demo
    prof_parser = sub.add_parser('profile', help='Profile model speed')
    prof_parser.add_argument('--width', type=int, default=32)
    prof_parser.add_argument('--input_size', type=int, default=256)

    # Realtime demo
    rt_parser = sub.add_parser('realtime', help='Real-time camera demo')
    rt_parser.add_argument('--model', default='model/hrnet_w32.pth')
    rt_parser.add_argument('--source', default=0, type=int, help='Camera index or video file path')
    rt_parser.add_argument('--width', type=int, default=32)
    rt_parser.add_argument('--input_size', type=int, default=256)

    args = parser.parse_args()
    # 如果没有提供任何子命令，默认使用 image 并设置所有必需参数
    if args.command is None:
        args.command = 'image'
        args.image = 'images/000000033221.jpg'
        args.model = 'checkpoints/hrnet_w32_epoch0013_loss0.0716.pth'
        args.width = 32
        args.input_size = 256


    if args.command == 'image':
        demo_image(args.image, args.model, args.width, args.input_size)
    elif args.command == 'profile':
        demo_profile(args.width, args.input_size)
    elif args.command == 'realtime':
        demo_realtime(args.model, args.source, args.width, args.input_size)
    else:
        parser.print_help()
