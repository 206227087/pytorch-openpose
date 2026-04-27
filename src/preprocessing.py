"""Shared image preprocessing utilities for OpenPose.

Used by both training (train.py) and inference (src/body.py, src/hand.py).
"""

import cv2
import numpy as np
import torch

from src.config import INPUT_SIZE, STRIDE, PAD_VALUE


def pad_image(img, stride=STRIDE, pad_value=PAD_VALUE):
    """Pad image on right and bottom to make dimensions divisible by stride.

    Returns:
        padded_img: Padded image
        pad: [pad_top, pad_left, pad_bottom, pad_right] values
    """
    h, w = img.shape[:2]
    pad = [0, 0, 0, 0]  # top, left, bottom, right

    if h % stride != 0:
        pad[2] = stride - (h % stride)
    if w % stride != 0:
        pad[3] = stride - (w % stride)

    padded = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                cv2.BORDER_CONSTANT, value=[pad_value] * img.shape[2]
        if img.ndim == 3 else pad_value)
    return padded, pad


def normalize_image(img):
    """Normalize image from uint8 [0,255] to float32 [-0.5, 0.5].

    Consistent normalization used by both training and inference.
    """
    return img.astype(np.float32) / 255.0 - 0.5


def prepare_model_input(img, scale, stride=STRIDE, pad_value=PAD_VALUE):
    """Full preprocessing pipeline for model inference.

    Args:
        img: Original image (H, W, 3) uint8
        scale: Resize scale factor
        stride: Model stride (default 8)
        pad_value: Padding value (default 128)

    Returns:
        data: Preprocessed tensor (1, 3, H', W') ready for model
        padded_shape: Shape of padded image before tensor conversion
        pad: Padding values [top, left, bottom, right]
    """
    # Resize
    resized = cv2.resize(img, (0, 0), fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)

    # Pad to multiple of stride
    padded, pad = pad_image(resized, stride, pad_value)

    # Normalize and convert to NCHW tensor
    normalized = normalize_image(padded)
    tensor = np.transpose(normalized[:, :, :, np.newaxis], (3, 2, 0, 1))
    tensor = np.ascontiguousarray(tensor)

    data = torch.from_numpy(tensor).float()
    if torch.cuda.is_available():
        data = data.cuda()

    return data, padded.shape, pad


def resize_output(output, stride, padded_shape, pad, original_shape):
    """Resize model output back to original image dimensions.

    Args:
        output: Model output array (H_out, W_out, C)
        stride: Model stride
        padded_shape: Shape of padded input image
        pad: Padding values [top, left, bottom, right]
        original_shape: Original image shape (H, W, C)

    Returns:
        Resized output matching original image dimensions
    """
    # Upsample by stride
    result = cv2.resize(output, (0, 0), fx=stride, fy=stride,
                        interpolation=cv2.INTER_CUBIC)

    # Remove padding
    # result = result[:padded_shape[0] - pad[2], :padded_shape[1] - pad[3], :]
    result = result[pad[0]:padded_shape[0] - pad[2], pad[1]:padded_shape[1] - pad[3], :]

    # Resize to original image size
    result = cv2.resize(result, (original_shape[1], original_shape[0]),
                        interpolation=cv2.INTER_CUBIC)
    return result
