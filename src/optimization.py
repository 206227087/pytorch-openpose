"""Model optimization utilities for lightweighting and acceleration.

Provides depthwise separable convolutions, channel pruning, and knowledge
distillation as described in the HRNet/OpenPose optimization guide.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Depthwise Separable Convolution ──────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: replaces standard Conv2d for efficiency.

    Factorizes a standard convolution into a depthwise conv (per-channel)
    followed by a pointwise conv (1x1 cross-channel), reducing FLOPs by
    approximately 1/out_channels + 1/kernel_size^2.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: spatial kernel size (default 3).
        stride: stride (default 1).
        padding: padding (default 1).
        bias: whether to use bias (default False).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


def replace_conv_with_depthwise(model, skip_layers=None):
    """Replace Conv2d layers with DepthwiseSeparableConv in a model.

    Skips layers whose names contain any string in skip_layers.

    Args:
        model: nn.Module to optimize.
        skip_layers: list of name substrings to skip (default: final prediction layers).

    Returns:
        Modified model (in-place).
    """
    if skip_layers is None:
        skip_layers = ['final_layer', 'conv6_2_CPM', 'conv6_5_CPM',
                       'Mconv7', 'Mconv14']

    for name, module in model.named_modules():
        if any(skip in name for skip in skip_layers):
            continue
        for attr_name in dir(module):
            child = getattr(module, attr_name, None)
            if isinstance(child, nn.Conv2d) and child.groups == 1:
                if child.kernel_size == (3, 3) or child.kernel_size == (7, 7):
                    dw = DepthwiseSeparableConv(
                        child.in_channels, child.out_channels,
                        kernel_size=child.kernel_size[0],
                        stride=child.stride[0],
                        padding=child.padding[0],
                        bias=child.bias is not None,
                    )
                    setattr(module, attr_name, dw)
    return model


# ─── Channel Pruning ─────────────────────────────────────────────────────────

def compute_channel_importance(model, dataloader, loss_fn, device='cuda'):
    """Compute per-channel importance scores based on gradient magnitude.

    Channels with small gradient magnitudes contribute less to the loss
    and are candidates for pruning.

    Args:
        model: nn.Module to analyze.
        dataloader: validation data loader.
        loss_fn: loss function.
        device: compute device.

    Returns:
        Dict mapping parameter names to per-channel importance arrays.
    """
    model.eval()
    importance = {}

    # Initialize importance accumulators for each Conv2d weight
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 4:
            importance[name] = torch.zeros(param.shape[0], device=device)

    # Accumulate gradient magnitudes
    for batch in dataloader:
        model.zero_grad()
        inputs = batch[0].to(device)
        targets = batch[1].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if name in importance:
                importance[name] += param.grad.abs().mean(dim=[1, 2, 3]).detach()

    # Average over batches
    n = len(dataloader)
    for name in importance:
        importance[name] /= n

    return importance


def prune_channels(model, importance, prune_ratio=0.1):
    """Prune least important channels from Conv2d layers.

    Creates a pruned copy of the model with reduced channel counts.

    Args:
        model: original nn.Module.
        importance: per-channel importance dict from compute_channel_importance.
        prune_ratio: fraction of channels to prune (default 0.1).

    Returns:
        Pruned model copy.
    """
    pruned_model = copy.deepcopy(model)

    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d) and name + '.weight' in importance:
            imp = importance[name + '.weight']
            n_prune = int(len(imp) * prune_ratio)
            if n_prune == 0:
                continue
            # Keep the most important channels
            keep_indices = torch.argsort(imp, descending=True)[:-n_prune]
            keep_indices, _ = torch.sort(keep_indices)

            # Prune output channels
            with torch.no_grad():
                module.weight.data = module.weight.data[keep_indices]
                if module.bias is not None:
                    module.bias.data = module.bias.data[keep_indices]
                module.out_channels = len(keep_indices)

    return pruned_model


# ─── Knowledge Distillation ──────────────────────────────────────────────────

class DistillationLoss(nn.Module):
    """Knowledge distillation loss combining hard target and soft target losses.

    L = alpha * L_hard + (1 - alpha) * L_soft

    L_hard: standard MSE/CE loss with ground truth
    L_soft: KL divergence between student and teacher softmax outputs

    Args:
        alpha: weight for hard target loss (default 0.7).
        temperature: softmax temperature for soft targets (default 4.0).
        hard_loss_fn: loss function for hard targets (default MSELoss).
    """

    def __init__(self, alpha=0.7, temperature=4.0, hard_loss_fn=None):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.hard_loss_fn = hard_loss_fn or nn.MSELoss()

    def forward(self, student_output, teacher_output, ground_truth):
        # Hard target loss
        hard_loss = self.hard_loss_fn(student_output, ground_truth)

        # Soft target loss (KL divergence)
        T = self.temperature
        student_soft = F.log_softmax(student_output / T, dim=1)
        teacher_soft = F.softmax(teacher_output / T, dim=1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)

        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss


def distill_train_step(student, teacher, optimizer, batch, criterion,
                       device='cuda'):
    """Single training step for knowledge distillation.

    Args:
        student: student model (being trained).
        teacher: teacher model (frozen).
        optimizer: optimizer for student.
        batch: (inputs, ground_truth) tuple.
        criterion: DistillationLoss instance.
        device: compute device.

    Returns:
        Loss value for this step.
    """
    student.train()
    teacher.eval()

    inputs, ground_truth = batch[0].to(device), batch[1].to(device)

    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        student_out = student(inputs)
        with torch.no_grad():
            teacher_out = teacher(inputs)
        loss = criterion(student_out, teacher_out, ground_truth)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
