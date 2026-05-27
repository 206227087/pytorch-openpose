"""HRNet multi-person body pose estimation demo.

Demonstrates single-image and real-time multi-person pose estimation using
the HRNet backbone with PAF+heatmap dual-branch output.

Usage:
  python demo_hrnet.py image --image images/demo.jpg --model checkpoints/best.pth
  python demo_hrnet.py profile
  python demo_hrnet.py realtime --model checkpoints/best.pth
"""

import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import torch

import util
from hrnet_body_pose import BodyHRNetPose
from models.hrnet_model import HRNet


def demo_image(image_path, model_path, width=32, input_size=256):
    """Run HRNet multi-person pose estimation on a single image."""
    body = BodyHRNetPose(model_path, width=width, input_size=input_size)

    oriImg = cv2.imread(image_path)
    if oriImg is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")


    candidate, subset = body(oriImg)

    # oriImg = cv2.resize(oriImg, (1080, 1080),
    #                     interpolation=cv2.INTER_LINEAR)

    # Convert to OpenPose format for drawing
    # candidate_op, subset_op = convert_to_openpose_format(candidate.copy(), subset)
    canvas = util.draw_bodypose(oriImg, candidate, subset)

    # Save result
    save_dir = '../output/demo'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, canvas)
    print(f"Result saved to {save_path}")

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
    """Profile HRNet model speed."""
    from src.config import NUM_JOINTS, NUM_LIMBS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"--- HRNet-W{width} Profiling ---")
    hrnet = HRNet(num_joints=NUM_JOINTS, num_limbs=NUM_LIMBS, width=width).to(device).eval()

    # Parameter count
    params_M = sum(p.numel() for p in hrnet.parameters()) / 1e6
    print(f"  Parameters: {params_M:.2f}M")

    # Inference speed benchmark
    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            hrnet(dummy)
    if device == 'cuda':
        torch.cuda.synchronize()
    # Benchmark
    n_runs = 50
    t0 = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            hrnet(dummy)
    if device == 'cuda':
        torch.cuda.synchronize()
    avg_ms = (time.time() - t0) / n_runs * 1000
    print(f"  Avg inference: {avg_ms:.1f} ms")
    print(f"  FPS: {1000 / avg_ms:.1f}")
    if device == 'cuda':
        print(f"  GPU memory: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")


def demo_realtime(model_path, source=0, width=32, input_size=256):
    """Run real-time HRNet multi-person pose estimation on video/camera."""
    body = BodyHRNetPose(model_path, width=width, input_size=input_size)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    print("Press 'q' to quit")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        candidate, subset = body(frame)
        fps = 1.0 / (time.time() - t0 + 1e-6)

        # candidate_op, subset_op = convert_to_openpose_format(candidate.copy(), subset)
        # canvas = util.draw_bodypose(frame, candidate_op, subset_op)
        canvas = util.draw_bodypose(frame, candidate, subset)

        cv2.putText(canvas, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
    img_parser.add_argument('--image', default='../images/demo.jpg', help='Input image path')
    img_parser.add_argument('--model', default='../checkpoints/best.pth', help='HRNet model weights')
    img_parser.add_argument('--width', type=int, default=32, help='HRNet width (32 or 48)')
    img_parser.add_argument('--input_size', type=int, default=256)

    # Profile demo
    prof_parser = sub.add_parser('profile', help='Profile model speed')
    prof_parser.add_argument('--width', type=int, default=32)
    prof_parser.add_argument('--input_size', type=int, default=256)

    # Realtime demo
    rt_parser = sub.add_parser('realtime', help='Real-time camera demo')
    rt_parser.add_argument('--model', default='../checkpoints/best.pth')
    rt_parser.add_argument('--source', default=0, type=int, help='Camera index or video file path')
    rt_parser.add_argument('--width', type=int, default=32)
    rt_parser.add_argument('--input_size', type=int, default=256)

    args = parser.parse_args()
    # 如果没有提供任何子命令，默认使用 image 并设置所有必需参数
    if args.command is None:
        args.command = 'image'
        args.image = '../images/demo.jpg'
        # args.model = '../model/hrnet_w48_epoch0116_loss-2.25737.pth'
        args.model = '../checkpoints/best.pth'
        args.width = 48
        args.input_size = 256

    if args.command == 'image':
        demo_image(args.image, args.model, args.width, args.input_size)
    elif args.command == 'profile':
        demo_profile(args.width, args.input_size)
    elif args.command == 'realtime':
        demo_realtime(args.model, args.source, args.width, args.input_size)
    else:
        parser.print_help()
