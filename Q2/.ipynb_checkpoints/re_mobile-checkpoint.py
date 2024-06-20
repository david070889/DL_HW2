import torch
import torch.nn as nn
import torch.nn.functional as F

class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale

class MobileNetV3BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, reduction=4, use_hs=True):
        super(MobileNetV3BasicBlock, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=(kernel_size-1)//2, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = HSwish() if use_hs else nn.ReLU(inplace=True)

        self.se = SqueezeExcitation(in_channels, reduction)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act2 = HSwish() if use_hs else nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.se(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_res_connect:
            return x + out
        else:
            return out


class CustomMobileNetV3(nn.Module):
    def __init__(self, num_classes=50):
        super(CustomMobileNetV3, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = HSwish()

        self.block1 = MobileNetV3BasicBlock(16, 24, kernel_size=3, stride=2)
        self.block2 = MobileNetV3BasicBlock(24, 24, kernel_size=3, stride=1)
        self.block3 = MobileNetV3BasicBlock(24, 40, kernel_size=5, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(40, 1280)
        self.hs2 = HSwish()
        self.fc2 = nn.Linear(1280, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.hs1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.hs2(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

