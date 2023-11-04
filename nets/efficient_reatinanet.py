import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.anchors import Anchors
from nets.vision_transformer import VisionTransformer

from nets.frepvgg import create_Rep

class ChangeChannel(nn.Module):
    def __init__(self,C4_size, C5_size, feature_size=256):
        super(ChangeChannel, self).__init__()
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        C4, C5 = x
        P4_x = self.P4_1(C4)
        P5_x = self.P5_1(C5)
        return [P4_x, P5_x]

class PyramidFeatures(nn.Module):
    def __init__(self, feature_size=256):
        super(PyramidFeatures, self).__init__()

        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_2 = nn.Conv2d(feature_size*2, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_2 = nn.Conv2d(feature_size*2, feature_size, kernel_size=3, stride=1, padding=1)

        self.P6 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        P3_x, P4_x, P5_x = inputs
        _, _, h4, w4 = P4_x.size()
        _, _, h3, w3 = P3_x.size()

        # 19,19,256 -> 38,38,256
        P5_upsampled_x = F.interpolate(P5_x, size=(h4, w4))
        # 38,38,256 cat 38,38,256 -> 38,38,512
        P4_x = torch.cat([P5_upsampled_x, P4_x], dim=1)
        # 38,38,512  ->  38,38,256
        P4_x = self.P4_2(P4_x)
        # 38,38,256 -> 75,75,256
        P4_upsampled_x = F.interpolate(P4_x, size=(h3, w3))
        # 75,75,256 cat 75,75,256 -> 75,75,512
        P3_x = torch.cat([P3_x, P4_upsampled_x], dim=1)

        # 75,75,512 -> 75,75,256
        P3_x = self.P3_2(P3_x)

        # 19,19,256 -> 19,19,256
        P5_x = self.P5_2(P5_x)

        # 19,19,256 -> 10,10,256
        P6_x = self.P6(P5_x)

        P7_x = self.P7_1(P6_x)
        # 10,10,256 -> 5,5,256
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1  = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1   = nn.ReLU()

        self.conv2  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2   = nn.ReLU()

        # self.conv3  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        # self.act3   = nn.ReLU()
        #
        # self.conv4  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        # self.act4   = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        # out = self.conv3(out)
        # out = self.act3(out)
        #
        # out = self.conv4(out)
        # out = self.act4(out)

        out = self.output(out)

        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, feature_size=128):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1  = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1   = nn.ReLU()

        self.conv2  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2   = nn.ReLU()

        # self.conv3  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        # self.act3   = nn.ReLU()
        #
        # self.conv4  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        # self.act4   = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        # out = self.conv3(out)
        # out = self.act3(out)
        #
        # out = self.conv4(out)
        # out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, height, width, channels = out1.shape

        out2 = out1.view(batch_size, height, width, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

class Repvgg(nn.Module):
    def __init__(self, pretrained=False, deploy=False):
        super(Repvgg, self).__init__()
        model = create_Rep(pretrained,deploy)
        self.model = model

    def forward(self, x):
        x = self.model.stage0(x)
        x = self.model.stage1(x)
        x = self.model.stage2(x)
        feat1 = x
        x = self.model.stage3(x)
        feat2 = x
        feat3 = self.model.stage4(x)

        return [feat1,feat2,feat3]

class ERetinaNet(nn.Module):
    def __init__(self, num_classes,  pretrained=False, deploy=False):
        super(ERetinaNet, self).__init__()
        self.pretrained = pretrained

        #   backbone output: C3,C4,C5
        self.backbone_net = Repvgg(pretrained, deploy)

        self.changeChannel = ChangeChannel(512, 2048)

        #   add ViT blocks   Note: If the input_shape is not [600, 600], the img_size here needs to change accordingly.
        self.vit_c4 = VisionTransformer(img_size=38, in_c=256, embed_dim=256, depth=2, num_heads=8,
                                      representation_size=None, num_classes=num_classes)
        self.vit_c5 = VisionTransformer(img_size=19, in_c=256, embed_dim=256, depth=2, num_heads=8,
                                      representation_size=None, num_classes=num_classes)

        self.fpn = PyramidFeatures()

        self.regressionModel        = RegressionModel(256)
        self.classificationModel    = ClassificationModel(256, num_classes=num_classes)
        self.anchors = Anchors()
        self._init_weights()

    def _init_weights(self):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        
        prior = 0.01
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

    def forward(self, inputs):
        p3, p4, p5 = self.backbone_net(inputs)
        px4, px5 = self.changeChannel([p4, p5])

        px4 = self.vit_c4(px4)
        px5 = self.vit_c5(px5)

        features = self.fpn([p3, px4, px5])

        regression      = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        classification  = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(features)

        return features, regression, classification, anchors

# model = retinanet(1)
# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# print(model.to(device))