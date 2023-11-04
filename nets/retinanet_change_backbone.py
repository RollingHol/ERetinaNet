import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.anchors import Anchors

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import vgg16_bn, inception_v3, mobilenet_v3_large

from nets.repvgg import create_RepVGG_B1g2
from nets.frepvgg import create_Rep

class ChangeChannel(nn.Module):
    def __init__(self,C3_size, C4_size, C5_size, feature_size=256):
        super(ChangeChannel, self).__init__()
        self.C3_c = C3_size
        self.out_c = feature_size
        if self.C3_c!=self.out_c:
            self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        if self.C3_c!=self.out_c:
            # 75,75,512 -> 75,75,256
            P3_x = self.P3_1(C3)
        else:
            P3_x = C3
        # 38,38,1024 -> 38,38,256
        P4_x = self.P4_1(C4)
        # 19,19,2048 -> 19,19,256
        P5_x = self.P5_1(C5)
        return [P3_x, P4_x, P5_x]

class PyramidFeatures(nn.Module):
    def __init__(self, feature_size=256):
        super(PyramidFeatures, self).__init__()

        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P6 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        P3_x, P4_x, P5_x = inputs
        _, _, h4, w4 = P4_x.size()
        _, _, h3, w3 = P3_x.size()

        # 19,19,256 -> 38,38,256
        P5_upsampled_x = F.interpolate(P5_x, size=(h4, w4))
        # 38,38,256 + 38,38,256 -> 38,38,256
        P4_x = P5_upsampled_x + P4_x
        # 38,38,256 -> 75,75,256
        P4_upsampled_x = F.interpolate(P4_x, size=(h3, w3))
        # 75,75,256 + 75,75,256 -> 75,75,256
        P3_x = P3_x + P4_upsampled_x

        # 75,75,256 -> 75,75,256
        P3_x = self.P3_2(P3_x)
        # 38,38,256 -> 38,38,256
        P4_x = self.P4_2(P4_x)
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

        self.conv3  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3   = nn.ReLU()

        self.conv4  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4   = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1  = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1   = nn.ReLU()

        self.conv2  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2   = nn.ReLU()

        self.conv3  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3   = nn.ReLU()

        self.conv4  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4   = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, height, width, channels = out1.shape

        out2 = out1.view(batch_size, height, width, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

class ResNet(nn.Module):
    def __init__(self, phi, pretrained=False):
        super(ResNet, self).__init__()
        self.edition = [resnet18, resnet34, resnet50, resnet101, resnet152]
        model = self.edition[phi](pretrained)
        del model.avgpool
        del model.fc
        self.model = model

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        feat1 = self.model.layer2(x)
        feat2 = self.model.layer3(feat1)
        feat3 = self.model.layer4(feat2)

        return [feat1, feat2, feat3]

class RepVGG(nn.Module):
    def __init__(self, pretrained=False, deploy=False):
        super(RepVGG, self).__init__()
        model = create_RepVGG_B1g2(pretrained,deploy)
        self.model = model

    def forward(self, x):
        x = self.model.stage0(x)
        for i in range(0,4):
            x = self.model.stage1[i](x)
        for i in range(0,6):
            x = self.model.stage2[i](x)
        feat1 = x
        for i in range(0,16):
            x = self.model.stage3[i](x)
        feat2 = x
        feat3 = self.model.stage4[0](x)

        return [feat1,feat2,feat3]

class FRepVGG(nn.Module):
    def __init__(self, pretrained=False, deploy=False):
        super(FRepVGG, self).__init__()
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

class VGG16(nn.Module):
    def __init__(self, pretrained=False):
        super(VGG16, self).__init__()
        model = vgg16_bn(pretrained)
        self.model = model

    def forward(self, x):
        for i in range(44):
            x = self.model.features[i](x)
            if i == 32:
                feat1 = x
            if i == 42:
                feat2 = x
            if i == 43:
                feat3 = x
        return [feat1, feat2, feat3]

class InceptionV3(nn.Module):
    def __init__(self, pretrained=False):
        super(InceptionV3, self).__init__()
        model = inception_v3(pretrained)
        self.model = model

    def forward(self, input):
        #   input:N, 3, 600, 600
        #   N, 32, 299, 299
        x = self.model.Conv2d_1a_3x3(input)
        #   N, 32, 297, 297
        x = self.model.Conv2d_2a_3x3(x)
        #   N, 64, 297, 297
        x = self.model.Conv2d_2b_3x3(x)
        #   N, 64, 148, 148
        x = self.model.maxpool1(x)
        #   N, 80, 148, 148
        x = self.model.Conv2d_3b_1x1(x)
        #   N, 192, 146, 146
        x = self.model.Conv2d_4a_3x3(x)
        #   N, 192, 72, 72
        x = self.model.maxpool2(x)
        #   N, 256, 72, 72
        x = self.model.Mixed_5b(x)
        #   N, 288, 72, 72
        x = self.model.Mixed_5c(x)
        #   N, 288, 72, 72
        x = self.model.Mixed_5d(x)    #√
        feat1 = x
        #   N, 768, 35, 35
        x = self.model.Mixed_6a(x)
        #   N, 768, 35, 35
        x = self.model.Mixed_6b(x)
        #   N, 768, 35, 35
        x = self.model.Mixed_6c(x)
        #   N, 768, 35, 35
        x = self.model.Mixed_6d(x)
        #   N, 768, 35, 35
        x = self.model.Mixed_6e(x)   #√
        feat2 = x
        #   N, 1280, 17, 17
        x = self.model.Mixed_7a(x)
        #   N, 2048, 17, 17
        x = self.model.Mixed_7b(x)
        #   N, 2048, 17, 17
        x = self.model.Mixed_7c(x)    #√
        feat3 = x

        return [feat1, feat2, feat3]

class MobileNet(nn.Module):
    def __init__(self, pretrained=False):
        super(MobileNet, self).__init__()
        model = mobilenet_v3_large(pretrained)
        self.model = model

    def forward(self, x):
        for i in range(16):
            x = self.model.features[i](x)
            if i == 6:
                feat1 = x
            if i == 12:
                feat2 = x
            if i == 15:
                feat3 = x
        return [feat1, feat2, feat3]

class retinanet(nn.Module):
    def __init__(self, num_classes, backbone, pretrained=False):
        super(retinanet, self).__init__()
        self.pretrained = pretrained

        #  C3、C4、C5
        if backbone.split("-")[0] == "resnet":
            phis = {"18":0, "34":1, "50":2, "101":3, "152":4}
            phi = phis[backbone.split("-")[1]]
            self.backbone_net = ResNet(phi, pretrained)
            backbone_outC = {
                0: [128, 256, 512],
                1: [128, 256, 512],
                2: [512, 1024, 2048],
                3: [512, 1024, 2048],
                4: [512, 1024, 2048],
            }[phi]
        elif backbone.split("-")[0] == "repvgg":
            deploy = False if backbone.split("-")[1] == "train" else True
            self.backbone_net = RepVGG(pretrained, deploy)
            backbone_outC = [256, 512, 2048]
        elif backbone.split("-")[0] == "frepvgg":
            deploy = False if backbone.split("-")[1] == "train" else True
            self.backbone_net = FRepVGG(pretrained, deploy)
            backbone_outC = [256, 512, 2048]
        elif backbone.split("-")[0] == "vgg":
            self.backbone_net = VGG16(pretrained)
            backbone_outC = [512, 512, 512]
        elif backbone.split("-")[0] == "inception":
            self.backbone_net = InceptionV3(pretrained)
            backbone_outC = [288, 768, 2048]
        elif backbone.split("-")[0] == "mobilenet":
            self.backbone_net = MobileNet(pretrained)
            backbone_outC = [40, 112, 160]
        else:
            raise ValueError("The backbone input is incorrect.")


        self.changeChannel = ChangeChannel(backbone_outC[0],backbone_outC[1],backbone_outC[2])
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
        px3, px4, px5 = self.changeChannel([p3, p4, p5])
        features = self.fpn([px3, px4, px5])

        #  prediction
        regression      = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        classification  = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(features)

        return features, regression, classification, anchors

# model = retinanet(1,2)
# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# print(model.to(device))