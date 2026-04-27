"""Neural network model definitions for OpenPose.

Contains bodypose_model (inference), BodyPoseTrainModel (training with
intermediate supervision), and handpose_model.
"""

from collections import OrderedDict

import torch
import torch.nn as nn


def make_layers(block, no_relu_layers):
    """Build nn.Sequential layers from an OrderedDict specification.

    Each entry is either a MaxPool2d (if 'pool' in name) or Conv2d.
    ReLU is appended unless the layer name is in no_relu_layers.
    """
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3], padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append((f'{layer_name}_relu', nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))


# --- Body Pose Model ---------------------------------------------------------

class bodypose_model(nn.Module):
    """OpenPose body pose network (inference mode).

    Returns the final stage output: (PAFs, heatmaps).
    Architecture: VGG feature extractor + 6 refinement stages,
    each with a PAF branch (38 ch) and heatmap branch (19 ch).
    """

    def __init__(self):
        super(bodypose_model, self).__init__()

        no_relu_layers = [
            'conv5_5_CPM_L1', 'conv5_5_CPM_L2',
            'Mconv7_stage2_L1', 'Mconv7_stage3_L1', 'Mconv7_stage4_L1',
            'Mconv7_stage5_L1', 'Mconv7_stage6_L1',
            'Mconv7_stage2_L2', 'Mconv7_stage3_L2', 'Mconv7_stage4_L2',
            'Mconv7_stage5_L2', 'Mconv7_stage6_L2',
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
        ])

        # Stage 1 - PAF branch: 3x3 conv -> 3x3 conv -> 3x3 conv -> 1x1 bottleneck -> 1x1 output (38 ch)
        block1_1 = OrderedDict([
            ('conv5_1_CPM_L1', [128, 128, 3, 1, 1]),
            ('conv5_2_CPM_L1', [128, 128, 3, 1, 1]),
            ('conv5_3_CPM_L1', [128, 128, 3, 1, 1]),
            ('conv5_4_CPM_L1', [128, 512, 1, 1, 0]),
            ('conv5_5_CPM_L1', [512, 38, 1, 1, 0]),
        ])

        # Stage 1 - Heatmap branch: same structure, output 19 ch (18 joints + bg)
        block1_2 = OrderedDict([
            ('conv5_1_CPM_L2', [128, 128, 3, 1, 1]),
            ('conv5_2_CPM_L2', [128, 128, 3, 1, 1]),
            ('conv5_3_CPM_L2', [128, 128, 3, 1, 1]),
            ('conv5_4_CPM_L2', [128, 512, 1, 1, 0]),
            ('conv5_5_CPM_L2', [512, 19, 1, 1, 0]),
        ])

        blocks = {
            'block0': block0,
            'block1_1': block1_1,
            'block1_2': block1_2,
        }

        # Stages 2-6: input is concat(prev_paf, prev_hm, features) = 38+19+128 = 185 ch
        for i in range(2, 7):
            blocks[f'block{i}_1'] = OrderedDict([
                (f'Mconv1_stage{i}_L1', [185, 128, 7, 1, 3]),
                (f'Mconv2_stage{i}_L1', [128, 128, 7, 1, 3]),
                (f'Mconv3_stage{i}_L1', [128, 128, 7, 1, 3]),
                (f'Mconv4_stage{i}_L1', [128, 128, 7, 1, 3]),
                (f'Mconv5_stage{i}_L1', [128, 128, 7, 1, 3]),
                (f'Mconv6_stage{i}_L1', [128, 128, 1, 1, 0]),
                (f'Mconv7_stage{i}_L1', [128, 38, 1, 1, 0]),
            ])
            blocks[f'block{i}_2'] = OrderedDict([
                (f'Mconv1_stage{i}_L2', [185, 128, 7, 1, 3]),
                (f'Mconv2_stage{i}_L2', [128, 128, 7, 1, 3]),
                (f'Mconv3_stage{i}_L2', [128, 128, 7, 1, 3]),
                (f'Mconv4_stage{i}_L2', [128, 128, 7, 1, 3]),
                (f'Mconv5_stage{i}_L2', [128, 128, 7, 1, 3]),
                (f'Mconv6_stage{i}_L2', [128, 128, 1, 1, 0]),
                (f'Mconv7_stage{i}_L2', [128, 19, 1, 1, 0]),
            ])

        for k in blocks:
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model0 = blocks['block0']
        self.model1_1 = blocks['block1_1']
        self.model1_2 = blocks['block1_2']
        self.model2_1 = blocks['block2_1']
        self.model2_2 = blocks['block2_2']
        self.model3_1 = blocks['block3_1']
        self.model3_2 = blocks['block3_2']
        self.model4_1 = blocks['block4_1']
        self.model4_2 = blocks['block4_2']
        self.model5_1 = blocks['block5_1']
        self.model5_2 = blocks['block5_2']
        self.model6_1 = blocks['block6_1']
        self.model6_2 = blocks['block6_2']

    def forward(self, x):
        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        return out6_1, out6_2


class BodyPoseTrainModel(bodypose_model):
    """Training wrapper that returns per-stage (PAF, heatmap) pairs.

    Used for intermediate supervision during training. The loss is
    computed at all 6 stages, not just the final one.
    """

    def forward(self, x):
        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        paf_stages = [out1_1, out2_1, out3_1, out4_1, out5_1, out6_1]
        hm_stages = [out1_2, out2_2, out3_2, out4_2, out5_2, out6_2]
        return paf_stages, hm_stages


# --- Hand Pose Model ---------------------------------------------------------

class handpose_model(nn.Module):
    """OpenPose hand pose network.

    Returns the final stage output (22-channel heatmap:
    21 hand keypoints + background).
    """

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
