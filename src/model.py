"""Neural network model definitions for OpenPose.

Contains bodypose_model (inference), BodyPoseTrainModel (training with
intermediate supervision), handpose_model, and HRNet backbone.
"""

from collections import OrderedDict

import torch
import torch.nn as nn


def make_layers(block, no_relu_layers):
    """Build nn.Sequential layers from an OrderedDict specification.

    Each entry is either a MaxPool2d (if 'pool' in name) or Conv2d.
    ReLU is appended unless the layer name is in no_relu_layers.
    """
    layers = nn.Sequential()
    for i, (key, v) in enumerate(block.items()):
        if 'pool' in key:
            # MaxPool2d: [kernel, stride, padding]
            layers.add_module(key, nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2]))
        else:
            # Conv2d: [in_ch, out_ch, kernel, stride, padding]
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3], padding=v[4])
            layers.add_module(key, conv2d)
            if key not in no_relu_layers:
                layers.add_module(f'{key}_relu', nn.ReLU(inplace=True))
    return layers


# --- OpenPose Body Model (Inference) -----------------------------------------

class bodypose_model(nn.Module):
    """OpenPose body pose estimation model for inference.

    Returns only the final stage PAF and heatmap outputs.
    Architecture: VGG-like feature extractor + 6 refinement stages,
    each with dual branches for PAF (38-ch) and heatmap (19-ch) prediction.
    """

    def __init__(self):
        super(bodypose_model, self).__init__()

        no_relu_layers = [
            'conv5_5_CPM', 'conv6_2_CPM',
            'Mconv7_stage2', 'Mconv7_stage3', 'Mconv7_stage4',
            'Mconv7_stage5', 'Mconv7_stage6',
        ]

        # Feature extraction (VGG-like front-end)
        block0 = OrderedDict([
            ('conv1_1', [3, 64, 3, 1, 1]),
            ('conv1_2', [64, 64, 3, 1, 1]),
            ('pool1_stage1', [2, 2, 0]),
            ('conv2_1', [64, 128, 3, 1, 1]),
            ('conv2_2', [128, 128, 3, 1, 1]),
            ('pool2_stage1', [2, 2, 0]),
            ('conv3_1', [128, 256, 3, 1, 1]),
            ('conv3_2', [256, 256, 3, 1, 1]),
            ('conv3_3', [256, 256, 3, 1, 1]),
            ('conv3_4', [256, 256, 3, 1, 1]),
            ('pool3_stage1', [2, 2, 0]),
            ('conv4_1', [256, 512, 3, 1, 1]),
            ('conv4_2', [512, 512, 3, 1, 1]),
            ('conv4_3_CPM', [512, 256, 3, 1, 1]),
            ('conv4_4_CPM', [256, 128, 3, 1, 1]),
            ('conv5_5_CPM', [128, 128, 3, 1, 1]),
        ])

        # Stage 1 - PAF branch
        block1_0 = OrderedDict([
            ('conv6_0_CPM', [128, 512, 1, 1, 0]),
            ('conv6_1_CPM', [512, 128, 1, 1, 0]),
            ('conv6_2_CPM', [128, 38, 1, 1, 0]),
        ])

        # Stage 1 - Heatmap branch
        block1_1 = OrderedDict([
            ('conv6_3_CPM', [128, 512, 1, 1, 0]),
            ('conv6_4_CPM', [512, 128, 1, 1, 0]),
            ('conv6_5_CPM', [128, 19, 1, 1, 0]),
        ])

        blocks = {}
        blocks['block0'] = block0
        blocks['block1_0'] = block1_0
        blocks['block1_1'] = block1_1

        # Stages 2-6
        for i in range(2, 7):
            blocks[f'block{i}_0'] = OrderedDict([
                (f'Mconv1_stage{i}', [185, 128, 7, 1, 3]),
                (f'Mconv2_stage{i}', [128, 128, 7, 1, 3]),
                (f'Mconv3_stage{i}', [128, 128, 7, 1, 3]),
                (f'Mconv4_stage{i}', [128, 128, 7, 1, 3]),
                (f'Mconv5_stage{i}', [128, 128, 7, 1, 3]),
                (f'Mconv6_stage{i}', [128, 128, 1, 1, 0]),
                (f'Mconv7_stage{i}', [128, 38, 1, 1, 0]),
            ])
            blocks[f'block{i}_1'] = OrderedDict([
                (f'Mconv8_stage{i}', [185, 128, 7, 1, 3]),
                (f'Mconv9_stage{i}', [128, 128, 7, 1, 3]),
                (f'Mconv10_stage{i}', [128, 128, 7, 1, 3]),
                (f'Mconv11_stage{i}', [128, 128, 7, 1, 3]),
                (f'Mconv12_stage{i}', [128, 128, 7, 1, 3]),
                (f'Mconv13_stage{i}', [128, 128, 1, 1, 0]),
                (f'Mconv14_stage{i}', [128, 19, 1, 1, 0]),
            ])

        for k in blocks:
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model0 = blocks['block0']
        self.model1_0 = blocks['block1_0']
        self.model1_1 = blocks['block1_1']
        self.model2_0 = blocks['block2_0']
        self.model2_1 = blocks['block2_1']
        self.model3_0 = blocks['block3_0']
        self.model3_1 = blocks['block3_1']
        self.model4_0 = blocks['block4_0']
        self.model4_1 = blocks['block4_1']
        self.model5_0 = blocks['block5_0']
        self.model5_1 = blocks['block5_1']
        self.model6_0 = blocks['block6_0']
        self.model6_1 = blocks['block6_1']

    def forward(self, x):
        out0 = self.model0(x)
        out1_0 = self.model1_0(out0)  # PAF stage 1
        out1_1 = self.model1_1(out0)  # HM stage 1

        concat2 = torch.cat([out0, out1_0, out1_1], 1)
        out2_0 = self.model2_0(concat2)  # PAF stage 2
        out2_1 = self.model2_1(concat2)  # HM stage 2

        concat3 = torch.cat([out0, out2_0, out2_1], 1)
        out3_0 = self.model3_0(concat3)
        out3_1 = self.model3_1(concat3)

        concat4 = torch.cat([out0, out3_0, out3_1], 1)
        out4_0 = self.model4_0(concat4)
        out4_1 = self.model4_1(concat4)

        concat5 = torch.cat([out0, out4_0, out4_1], 1)
        out5_0 = self.model5_0(concat5)
        out5_1 = self.model5_1(concat5)

        concat6 = torch.cat([out0, out5_0, out5_1], 1)
        out6_0 = self.model6_0(concat6)
        out6_1 = self.model6_1(concat6)

        return out6_0, out6_1


# --- OpenPose Body Model (Training) ------------------------------------------

class BodyPoseTrainModel(bodypose_model):
    """Training variant that returns all 6 stages for intermediate supervision."""

    def forward(self, x):
        out0 = self.model0(x)
        out1_0 = self.model1_0(out0)
        out1_1 = self.model1_1(out0)

        paf_stages = [out1_0]
        hm_stages = [out1_1]

        prev_paf, prev_hm = out1_0, out1_1
        for stage_i, (paf_mod, hm_mod) in enumerate([
            (self.model2_0, self.model2_1),
            (self.model3_0, self.model3_1),
            (self.model4_0, self.model4_1),
            (self.model5_0, self.model5_1),
            (self.model6_0, self.model6_1),
        ], start=2):
            concat = torch.cat([out0, prev_paf, prev_hm], 1)
            prev_paf = paf_mod(concat)
            prev_hm = hm_mod(concat)
            paf_stages.append(prev_paf)
            hm_stages.append(prev_hm)

        return paf_stages, hm_stages


# --- OpenPose Hand Model -----------------------------------------------------

class handpose_model(nn.Module):
    """Hand keypoint estimation model (21 keypoints + background)."""

    def __init__(self):
        super(handpose_model, self).__init__()

        no_relu_layers = [
            'conv6_2_CPM', 'Mconv7_stage2', 'Mconv7_stage3',
            'Mconv7_stage4', 'Mconv7_stage5', 'Mconv7_stage6',
        ]

        # Feature extraction
        block1_0 = OrderedDict([
            ('conv1_1', [3, 64, 3, 1, 1]),
            ('conv1_2', [64, 64, 3, 1, 1]),
            ('pool1_stage1', [2, 2, 0]),
            ('conv2_1', [64, 128, 3, 1, 1]),
            ('conv2_2', [128, 128, 3, 1, 1]),
            ('pool2_stage1', [2, 2, 0]),
            ('conv3_1', [128, 256, 3, 1, 1]),
            ('conv3_2', [256, 256, 3, 1, 1]),
            ('conv3_3', [256, 256, 3, 1, 1]),
            ('conv3_4', [256, 256, 3, 1, 1]),
            ('pool3_stage1', [2, 2, 0]),
            ('conv4_1', [256, 512, 3, 1, 1]),
            ('conv4_2', [512, 512, 3, 1, 1]),
            ('conv4_3', [512, 512, 3, 1, 1]),
            ('conv4_4', [512, 512, 3, 1, 1]),
            ('conv5_1', [512, 512, 3, 1, 1]),
            ('conv5_2', [512, 512, 3, 1, 1]),
            ('conv5_3_CPM', [512, 128, 3, 1, 1]),
        ])

        # Stage 1
        block1_1 = OrderedDict([
            ('conv6_1_CPM', [128, 512, 1, 1, 0]),
            ('conv6_2_CPM', [512, 22, 1, 1, 0]),
        ])

        blocks = {
            'block1_0': block1_0,
            'block1_1': block1_1,
        }

        # Stages 2-6
        for i in range(2, 7):
            blocks[f'block{i}'] = OrderedDict([
                (f'Mconv1_stage{i}', [150, 128, 7, 1, 3]),
                (f'Mconv2_stage{i}', [128, 128, 7, 1, 3]),
                (f'Mconv3_stage{i}', [128, 128, 7, 1, 3]),
                (f'Mconv4_stage{i}', [128, 128, 7, 1, 3]),
                (f'Mconv5_stage{i}', [128, 128, 7, 1, 3]),
                (f'Mconv6_stage{i}', [128, 128, 1, 1, 0]),
                (f'Mconv7_stage{i}', [128, 22, 1, 1, 0]),
            ])

        for k in blocks:
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_0 = blocks['block1_0']
        self.model1_1 = blocks['block1_1']
        self.model2 = blocks['block2']
        self.model3 = blocks['block3']
        self.model4 = blocks['block4']
        self.model5 = blocks['block5']
        self.model6 = blocks['block6']

    def forward(self, x):
        out1_0 = self.model1_0(x)
        out1_1 = self.model1_1(out1_0)
        concat_stage2 = torch.cat([out1_1, out1_0], 1)
        out_stage2 = self.model2(concat_stage2)
        concat_stage3 = torch.cat([out_stage2, out1_0], 1)
        out_stage3 = self.model3(concat_stage3)
        concat_stage4 = torch.cat([out_stage3, out1_0], 1)
        out_stage4 = self.model4(concat_stage4)
        concat_stage5 = torch.cat([out_stage4, out1_0], 1)
        out_stage5 = self.model5(concat_stage5)
        concat_stage6 = torch.cat([out_stage5, out1_0], 1)
        out_stage6 = self.model6(concat_stage6)
        return out_stage6


# --- HRNet: High-Resolution Network for Pose Estimation -----------------------
# Maintains parallel high-resolution feature streams to preserve spatial detail.
# Reference: "Deep High-Resolution Representation Learning for Visual Recognition"


class HRModule(nn.Module):
    """HRNet basic module: multi-resolution parallel convolutions with cross-branch fusion.

    Args:
        channels: list of channel counts for each resolution branch.
        num_blocks: number of basic conv blocks per branch.
    """

    def __init__(self, channels, num_blocks=1):
        super().__init__()
        num_branches = len(channels)

        # Branch-internal convolutions (repeated basic blocks)
        self.branches = nn.ModuleList()
        for c in channels:
            layers = []
            for _ in range(num_blocks):
                layers.append(nn.Sequential(
                    nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(c),
                ))
            self.branches.append(nn.Sequential(*layers))

        # Cross-branch fusion layers
        # branches[0] = highest resolution, branches[-1] = lowest resolution
        # fuse_layers[i][j]: transform from branch j to branch i's resolution
        self.fuse_layers = nn.ModuleList()
        for i in range(num_branches):
            fuse_ops = nn.ModuleList()
            for j in range(num_branches):
                if i == j:
                    fuse_ops.append(nn.Identity())
                elif i < j:
                    # Upsample from lower resolution branch j to higher resolution branch i
                    fuse_ops.append(nn.Sequential(
                        nn.Conv2d(channels[j], channels[i], 1, bias=False),
                        nn.BatchNorm2d(channels[i]),
                        nn.Upsample(scale_factor=2 ** (j - i)),
                    ))
                else:
                    # Downsample from higher resolution branch j to lower resolution branch i
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv2d(channels[j], channels[j], 3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(channels[j]),
                            nn.ReLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv2d(channels[j], channels[i], 3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(channels[i]),
                    ))
                    fuse_ops.append(nn.Sequential(*ops))
            self.fuse_layers.append(fuse_ops)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: list of multi-resolution feature maps [high_res, ..., low_res]
        branch_outs = [b(xi) for b, xi in zip(self.branches, x)]

        fused = []
        for i in range(len(branch_outs)):
            sum_feat = sum(
                self.fuse_layers[i][j](branch_outs[j])
                for j in range(len(branch_outs))
            )
            fused.append(self.relu(sum_feat + branch_outs[i]))
        return fused


class HRNet(nn.Module):
    """HRNet for multi-person pose estimation.

    Maintains high-resolution representations through parallel multi-resolution
    streams with repeated information exchange. Outputs both heatmaps and
    Part Affinity Fields (PAF) for multi-person grouping, matching OpenPose's
    dual-branch architecture.

    Args:
        num_joints: number of output keypoints (default 17 for COCO).
        num_limbs: number of PAF limb connections (default 16 for COCO).
        width: base channel width (e.g. 32 for HRNet-W32, 48 for HRNet-W48).
    """

    def __init__(self, num_joints=17, num_limbs=16, width=32):
        super().__init__()
        self.num_joints = num_joints
        self.num_limbs = num_limbs
        num_paf_channels = num_limbs * 2  # x, y per limb

        # Stem: initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 1: single branch
        self.stage1 = nn.Sequential(
            nn.Conv2d(64, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )

        # Transition 1->2: create second branch
        self.transition1 = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
        )

        # Stage 2: two resolution branches
        self.stage2 = HRModule([width, width * 2], num_blocks=1)

        # Transition 2->3: create third branch
        self.transition2 = nn.Sequential(
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
        )

        # Stage 3: three resolution branches
        self.stage3 = HRModule([width, width * 2, width * 4], num_blocks=1)

        # Transition 3->4: create fourth branch
        self.transition3 = nn.Sequential(
            nn.Conv2d(width * 4, width * 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 8),
            nn.ReLU(inplace=True),
        )

        # Stage 4: four resolution branches
        self.stage4 = HRModule([width, width * 2, width * 4, width * 8], num_blocks=1)

        # Dual prediction heads from highest resolution branch
        # Heatmap head: predicts num_joints channel heatmap
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, num_joints, kernel_size=1)
        )
        # PAF head: predicts num_limbs*2 channel part affinity fields
        self.paf_head = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, num_paf_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.stem(x)

        # Stage 1
        x1 = self.stage1(x)

        # Stage 2
        x2 = self.transition1(x1)
        x1, x2 = self.stage2([x1, x2])

        # Stage 3
        x3 = self.transition2(x2)
        x1, x2, x3 = self.stage3([x1, x2, x3])

        # Stage 4
        x4 = self.transition3(x3)
        x1, x2, x3, x4 = self.stage4([x1, x2, x3, x4])

        # Dual-branch output from highest resolution branch
        heatmap = self.heatmap_head(x1)  # (B, num_joints, H, W)
        paf = self.paf_head(x1)          # (B, num_limbs*2, H, W)
        return paf, heatmap


class HRNetTrainModel(HRNet):
    """Training variant of HRNet that returns single-stage output for loss.

    Unlike OpenPose's 6-stage intermediate supervision, HRNet uses a
    single-stage output. This class exists for API compatibility with
    the training loop which expects (paf_output, hm_output) format.
    """

    def forward(self, x):
        paf, heatmap = super().forward(x)
        return paf, heatmap
