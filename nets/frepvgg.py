# --------------------------------------------------------
# RepVGG: Making VGG-style ConvNets Great Again (https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)
# Github source: https://github.com/DingXiaoH/RepVGG
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import torch.nn as nn
import numpy as np
import torch
import copy

from nets.emca import MultiSpectralAttentionLayer as eMCA

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class ConvBnRule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvBnRule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.nonlinearity(x)
        return x

#RepDownBlock(first layer of each stage)、RepBlock
class RepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_att=False):
        super(RepBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        c2wh = dict([(64, 112), (128, 56), (256, 28), (512, 14), (2048, 7)])
        assert kernel_size == 3
        assert padding == 1

        # 0
        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_att:
            self.att = eMCA(out_channels, c2wh[out_channels], c2wh[out_channels])
        else:
            self.att = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.att(self.nonlinearity(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.att(self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

class RepStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, deploy=False, use_att=False):
        super(RepStage, self).__init__()
        self.deploy = deploy
        self.half_channels = out_channels // 2
        self.down = RepBlock(in_channels, out_channels, kernel_size=3, stride=2, padding=1, deploy=self.deploy)
        self.cbr1 = ConvBnRule(out_channels, self.half_channels, kernel_size=1)
        self.cbr2 = ConvBnRule(out_channels, self.half_channels, kernel_size=1)
        self.repBlocks = self._make_blocks(self.half_channels, num_blocks)
        self.fuse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(int((num_blocks+1)*self.half_channels), self.half_channels, kernel_size=1),
            nn.ReLU(),
            nn.Sigmoid(),
        )
        self.cbr3 = ConvBnRule(out_channels, out_channels, kernel_size=1)
        c2wh = dict([(64, 112), (128, 56), (256, 28), (512, 14), (2048, 7)])
        if use_att:
            self.att = eMCA(out_channels, c2wh[out_channels], c2wh[out_channels])
        else:
            self.att = nn.Identity()

    def _make_blocks(self, planes, num_blocks):
        blocks = []
        for i in range(0,num_blocks):
            blocks.append(RepBlock(planes, planes, kernel_size=3, stride=1, padding=1, deploy=self.deploy))
        return nn.ModuleList(blocks)

    def forward(self, x):
        x = self.down(x)
        #left branch
        x1 = self.cbr1(x)
        #right branch
        x2 = self.cbr2(x)
        e = x2

        eList = [x2]
        for block in self.repBlocks:
            e = block(e)
            eList.append(e)

        f = torch.cat(eList,1)
        f = self.fuse(f)
        f = f.view(-1, self.half_channels, 1, 1)
        out = x1 * f
        out = torch.cat((e,out),1)
        out = self.att(self.cbr3(out))
        return out

class RepBackbone(nn.Module):
    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, deploy=False, use_att=False):
        super(RepBackbone, self).__init__()
        assert len(width_multiplier) == 4
        self.deploy = deploy
        self.use_att = use_att

        self.stage0 = RepBlock(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, deploy=self.deploy)
        self.stage1 = RepStage(in_channels=64, out_channels=int(64*width_multiplier[0]), num_blocks=num_blocks[0], deploy=self.deploy, use_att=self.use_att)
        self.stage2 = RepStage(in_channels=int(64*width_multiplier[0]), out_channels=int(128*width_multiplier[1]), num_blocks=num_blocks[1], deploy=self.deploy, use_att=self.use_att)
        self.stage3 = RepStage(in_channels=int(128*width_multiplier[1]), out_channels=int(256*width_multiplier[2]), num_blocks=num_blocks[2], deploy=self.deploy, use_att=self.use_att)
        self.stage4 = RepBlock(in_channels=int(256*width_multiplier[2]), out_channels=int(512*width_multiplier[3]), kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_att=self.use_att)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(int(512*width_multiplier[3]), num_classes)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def create_Rep(pretrained=False, deploy=False, use_att=True, **kwargs):
    model = RepBackbone(num_blocks=[3,5,15], num_classes=100, width_multiplier=[2,2,2,4], deploy=deploy, use_att=use_att, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load('model_data/Faster_RepVGG-mini.pth'))
    return model


#   Use this for converting a RepVGG model or a bigger model with RepVGG as its component
#   Use like this
#   model = create_RepVGG_A0(deploy=False)
#   train model or load weights
#   repvgg_model_convert(model, save_path='repvgg_deploy.pth')
#   If you want to preserve the original model, call with do_copy=True

#   ====================== for using RepVGG as the backbone of a bigger model, e.g., PSPNet, the pseudo code will be like
#   train_backbone = create_RepVGG_B2(deploy=False)
#   train_backbone.load_state_dict(torch.load('RepVGG-B2-train.pth'))
#   train_pspnet = build_pspnet(backbone=train_backbone)
#   segmentation_train(train_pspnet)
#   deploy_pspnet = repvgg_model_convert(train_pspnet)
#   segmentation_test(deploy_pspnet)
#   =====================   example_pspnet.py shows an example

def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model

# model = create_Rep(deploy=True)
# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# print(model.to(device))