"""Neural network model definitions.

HRNet backbone for multiple body pose_model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# --- HRNet: High-Resolution Network for Pose Estimation -----------------------
# Maintains parallel high-resolution feature streams to preserve spatial detail.
# Reference: "Deep High-Resolution Representation Learning for Visual Recognition"


class Bottleneck(nn.Module):
    """Bottleneck block: 1x1 reduce -> 3x3 -> 1x1 expand with residual connection.

    Args:
        channels: input/output channel count.
        expansion: internal channel multiplier (default 1).
            Internal channels = channels // expansion for the 3x3 conv.
    """

    def __init__(self, channels, expansion=1):
        super().__init__()
        inner = channels // expansion
        self.reduce = nn.Conv2d(channels, inner, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inner)
        self.conv3x3 = nn.Conv2d(inner, inner, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inner)
        self.expand = nn.Conv2d(inner, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.reduce(x)))
        out = self.relu(self.bn2(self.conv3x3(out)))
        out = self.bn3(self.expand(out))
        return self.relu(out + identity)


class HRModule(nn.Module):
    """HRNet basic module: multi-resolution parallel convolutions with cross-branch fusion.

    Args:
        channels: list of channel counts for each resolution branch.
        num_blocks: number of Bottleneck blocks per branch.
    """

    def __init__(self, channels, num_blocks=1):
        super().__init__()
        num_branches = len(channels)

        # Branch-internal convolutions (repeated Bottleneck blocks)
        self.branches = nn.ModuleList()
        for c in channels:
            layers = []
            for _ in range(num_blocks):
                layers.append(Bottleneck(c))
            self.branches.append(nn.Sequential(*layers))

        # Cross-branch fusion layers
        # branches[0] = highest resolution, branches[-1] = lowest resolution
        # fuse_layers[i][j]: transform from branch j to branch i's resolution
        self.fuse_layers = nn.ModuleList()
        for i in range(num_branches):
            fuse_ops = nn.ModuleList()
            for j in range(num_branches):
                if i == j:
                    fuse_ops.append(None)
                elif i < j:
                    # Upsample from lower resolution branch j to higher resolution branch i
                    # Upsampling is done dynamically in forward() to handle odd sizes
                    fuse_ops.append(nn.Sequential(
                        nn.Conv2d(channels[j], channels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(channels[i]),
                    ))
                else:
                    # Downsample from higher resolution branch j to lower resolution branch i
                    # No ReLU between successive strided convolutions (per original HRNet paper)
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv2d(channels[j], channels[j], 3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(channels[j]),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv2d(channels[j], channels[i], 3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(channels[i]),
                    ))
                    fuse_ops.append(nn.Sequential(*ops))
            self.fuse_layers.append(fuse_ops)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # First apply branch-specific convolutions
        x_out = [self.branches[i](x[i]) for i in range(len(self.branches))]

        # Then fuse across branches
        x_fused = []
        for i in range(len(x_out)):
            # Initialize with branch 0 contribution
            if self.fuse_layers[i][0] is None:
                y = x_out[0]
            else:
                y = self.fuse_layers[i][0](x_out[0])

            # Add contributions from all other branches
            for j in range(1, len(x_out)):
                if i == j:
                    y = y + x_out[j]
                elif j > i:
                    # Lower resolution -> upsample to match target size
                    up = self.fuse_layers[i][j](x_out[j])
                    y = y + F.interpolate(up, size=y.shape[2:], mode='nearest')
                else:
                    # Higher resolution -> downsample
                    y = y + self.fuse_layers[i][j](x_out[j])

            x_fused.append(self.relu(y))

        return x_fused


class HRNet(nn.Module):
    """HRNet for multi-person pose estimation.

    Maintains high-resolution representations through parallel multi-resolution
    streams with repeated information exchange. Outputs both heatmaps and
    Part Affinity Fields (PAF) for multi-person grouping, matching OpenPose's
    dual-branch architecture.

    Args:
        num_joints: number of output key points (default 18).
        num_limbs: number of PAF limb connections (default 18).
        width: base channel width (e.g. 32 for HRNet-W32, 48 for HRNet-W48).
        dropout: dropout probability for prediction heads (default 0.0).
    """

    def __init__(self, num_joints=18, num_limbs=18, width=32, dropout=0.0):
        super().__init__()
        self.num_joints = num_joints
        self.num_limbs = num_limbs
        self.width = width
        self.dropout = dropout
        num_paf_channels = num_limbs * 2  # x, y per limb

        # Stem: initial feature extraction (channels aligned with width)
        stem_out = width // 2  # e.g. 16 for width=32, 24 for width=48
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_out, stem_out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_out),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 4 Bottleneck blocks at 1/4 resolution (standard HRNet)
        # Each Bottleneck: 1x1 -> 3x3 -> 1x1 conv with residual connection
        stage1_blocks = []
        for i in range(4):
            in_ch = stem_out if i == 0 else width
            stage1_blocks.append(BottleneckStage1(in_ch, width))
        self.stage1 = nn.Sequential(*stage1_blocks)

        # Transition 1->2: create second branch at 1/8 resolution
        self.transition1 = nn.ModuleList([
            None,  # Branch 0: keep as is
            nn.Sequential(
                nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(width * 2),
                nn.ReLU(inplace=True),
            )
        ])
        # Source branch indices for transition1 (-1 = identity, no downsample)
        self.register_buffer('transition1_src', torch.tensor([-1, 0]))

        # Stage 2: two resolution branches
        self.stage2 = HRModule([width, width * 2], num_blocks=4)

        # Transition 2->3: create third branch at 1/16 resolution
        # New branch is downsampled from the adjacent higher-resolution branch (branch 1)
        self.transition2 = nn.ModuleList([
            None,  # Branch 0: Identity
            None,  # Branch 1: Identity
            nn.Sequential(
                nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(width * 4),
                nn.ReLU(inplace=True)
            )  # Branch 2: Downsample from Branch 1
        ])
        self.register_buffer('transition2_src', torch.tensor([-1, -1, 1]))

        # Stage 3: three resolution branches
        self.stage3 = HRModule([width, width * 2, width * 4], num_blocks=4)

        # Transition 3->4: create fourth branch at 1/32 resolution
        # New branch is downsampled from the adjacent higher-resolution branch (branch 2)
        self.transition3 = nn.ModuleList([
            None,  # Branch 0: Identity
            None,  # Branch 1: Identity
            None,  # Branch 2: Identity
            nn.Sequential(
                nn.Conv2d(width * 4, width * 8, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(width * 8),
                nn.ReLU(inplace=True),
            )  # Branch 3: Downsample from Branch 2
        ])
        self.register_buffer('transition3_src', torch.tensor([-1, -1, -1, 2]))

        # Stage 4: four resolution branches
        self.stage4 = HRModule([width, width * 2, width * 4, width * 8], num_blocks=3)

        # Aggregate features from all resolution branches to highest resolution
        # Upsample lower-resolution branches and project to width channels
        self.aggregate = nn.ModuleList()
        branch_channels = [width, width * 2, width * 4, width * 8]
        for i, c in enumerate(branch_channels):
            if i == 0:
                self.aggregate.append(None)  # Already at highest resolution
            else:
                self.aggregate.append(nn.Sequential(
                    nn.Conv2d(c, width, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                ))

        # Dual prediction heads from aggregated features
        # Heatmap head: predicts num_joints channel heatmap (4 conv layers)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, num_joints, kernel_size=1)
        )
        # PAF head: predicts num_limbs*2 channel part affinity fields (4 conv layers)
        self.paf_head = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, num_paf_channels, kernel_size=1)
        )

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def _apply_transition(self, x_list, transition, transition_src):
        """Apply transition layer to prepare inputs for the next stage.

        Args:
            x_list: current list of branch outputs.
            transition: ModuleList of transition operations (None = identity).
            transition_src: tensor of source branch indices (-1 = identity).
        """
        x_list_next = []
        for i in range(len(transition)):
            src_idx = transition_src[i].item()
            if transition[i] is not None and src_idx >= 0:
                # New branch: downsample from the specified source branch
                x_list_next.append(transition[i](x_list[src_idx]))
            else:
                # Existing branch: keep as is
                x_list_next.append(x_list[i])
        return x_list_next

    def forward(self, x):
        x = self.stem(x)

        # Stage 1: process with single branch
        x1 = self.stage1(x)

        # Transition 1->2: prepare inputs for stage2
        x_list = self._apply_transition([x1], self.transition1, self.transition1_src)

        # Stage 2: two branches
        x_list = self.stage2(x_list)

        # Transition 2->3: prepare inputs for stage3
        x_list = self._apply_transition(x_list, self.transition2, self.transition2_src)

        # Stage 3: three branches
        x_list = self.stage3(x_list)

        # Transition 3->4: prepare inputs for stage4
        x_list = self._apply_transition(x_list, self.transition3, self.transition3_src)

        # Stage 4: four branches
        x_list = self.stage4(x_list)

        # Aggregate features from all resolution branches
        x_high_res = x_list[0]
        target_size = x_list[0].shape[2:]
        for i in range(1, len(x_list)):
            if self.aggregate[i] is not None:
                # Upsample to highest resolution and project
                upsampled = F.interpolate(
                    x_list[i], size=target_size, mode='nearest'
                )
                x_high_res = x_high_res + self.aggregate[i](upsampled)

        heatmap = torch.sigmoid(self.heatmap_head(x_high_res))  # [0, 1]
        paf = torch.tanh(self.paf_head(x_high_res))            # [-1, 1]
        return paf, heatmap

    def extra_repr(self):
        """Return extra representation string for print()."""
        return (f'num_joints={self.num_joints}, num_limbs={self.num_limbs}, '
                f'width={self.width}, dropout={self.dropout}')


class BottleneckStage1(nn.Module):
    """Bottleneck block for Stage 1 with optional input channel change and residual.

    Handles the in_ch -> width channel transition in the first block,
    and provides a residual connection (with 1x1 projection if channels differ).

    Args:
        in_channels: input channel count.
        out_channels: output channel count.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.expand = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Residual projection if input/output channels differ
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.reduce(x)))
        out = self.relu(self.bn2(self.conv3x3(out)))
        out = self.bn3(self.expand(out))
        return self.relu(out + identity)
