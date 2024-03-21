import torch
import torch.nn as nn
import torchvision


# 公用的 stem 层，对输入图像进行的第一次卷积操作。
# 卷积核尺寸： 7×7 ，卷积核个数 64，填充值为 3， 步长为 2
# 最大池化核： 3×3 ， 填充值为 1， 步长为 2
def Conv1(in_planes, places, stride=2):
    # [224, 224, 3]
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        # [112, 112, 64]
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # [56, 56, 64]
    )


# 残差块
# expansion = 4 为啥？ 因为我们在上面的图中可以发现统一的规律：经过三层的卷积运算，输出结果的通道数是输入通道数的4倍。
# downsampling 用来判断是否需要进行数据结构转换，就是上面说的 F(x) = F(x) + x ，两个数据结构不一致不能相加。

class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            # 第一个卷积层：卷积核尺寸： 1×1 ，填充值为 0， 步长为 1
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            # 第二个卷积层：卷积核尺寸： 3×3 ，填充值为 1， 步长为 1
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            # 第三个卷积层：卷积核尺寸： 1×1 ，填充值为 0， 步长为 1
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        # 判断 x 的数据格式（维度）是否和 F(x)的一样，如果不一样，则进行一次卷积运算，实现升维操作。
        # 卷积核尺寸： 1×1 ，个数为原始特征图通道数的4倍， 填充值为 0， 步长为 1
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    # 建立网络层运算结果的前向传递关系
    def forward(self, x):
        # 将输入赋值给 residual
        residual = x

        # 基础模块的运算结果
        out = self.bottleneck(x)

        # 如果x 的数据格式（维度）是和 F(x)不一样，则进行一次卷积运算，并将计算结果更新至residual
        if self.downsampling:
            residual = self.downsample(x)

        # 最终结果的求和（汇总）
        # 一部分是经过三层卷积运算的结果，另一部分是输入特征/输入特征的升维结果
        # out =  out + redidual
        out += residual

        out = self.relu(out)
        return out


# 构建一个ResNet类，内部调用上面的 stem 层和类 Bottleneck
# 参数 blocks 列表为50/101/152层中各 stage 中残差块的个数
class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=1000, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion

        # 调用 stem 层
        self.conv1 = Conv1(in_planes=3, places=64)

        # stage 1-4
        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

        # 均值池化
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # 全连接层
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))

        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
