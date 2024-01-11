import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, cfg[0], kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.conv2 = nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv3 = nn.Conv2d(cfg[1], cfg[2], kernel_size=1, stride=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(cfg[2])
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_classes=1000, cfg=None):
        super(ResNet, self).__init__()

        self.inplanes = cfg[0]

        #TODO: 這裡因為 sign_mnist 改成 1 channel ，之後如果需要再改回來
        #TODO: 因為 sign_mnist 的圖片太小， kernel_size=7 -> kernel_size=3 、 stride=2 -> stride=1 、 padding=3 -> padding=1 ，之後如果需要再改回來
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, cfg[1][-1], cfg[1])
        self.layer2 = self._make_layer(block, cfg[2][-1], cfg[2], stride=2)
        self.layer3 = self._make_layer(block, cfg[3][-1], cfg[3], stride=2)
        self.layer4 = self._make_layer(block, cfg[4][-1], cfg[4], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(cfg[4][-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 0.5)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(planes)
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0:3], stride, downsample))
        self.inplanes = planes
        for i in range(1, len(cfg)//3):
            layers.append(block(self.inplanes, planes, cfg[3*i: 3*(i+1)]))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
def resnet50(cfg=[64]+[64, 64, 256]*3+[128, 128, 512]*4+[256, 256, 1024]*6+[512, 512, 2048]*3):
    cfg = [cfg[0], cfg[1:10], cfg[10:22], cfg[22:40], cfg[40:]]
    return ResNet(Bottleneck, num_classes=26, cfg=cfg)