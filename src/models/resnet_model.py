import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation模块
    """
    def __init__(self, in_channels, reduction_ratio=16):
        """
        初始化Squeeze-and-Excitation模块
        Args:
            in_channels: 输入通道数
            reduction_ratio: 降维比,默认为16
        """
        super(SEBlock, self).__init__()
        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 全局平均池化
        se_weights = self.avg_pool(x).view(x.size(0), x.size(1))
        # 全连接层
        se_weights = self.fc(se_weights).view(x.size(0), x.size(1), 1, 1)
        # 通配符扩展
        se_weights = se_weights.expand_as(x)
        # 与输入张量相乘
        return x * se_weights


class ChannelAttention(nn.Module):
    """
    通道注意力模块
    """
    def __init__(self, in_channels, reduction_ratio=16):
        """
        初始化通道注意力模块
        Args:
            in_channels: 输入通道数
            reduction_ratio: 降维比,默认为16
        """
        super(ChannelAttention, self).__init__()
        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 全局平均池化和最大池化
        out = self.avg_pool(x).view(x.size(0), -1)
        # 全连接层
        out = self.fc(out).view(x.size(0), -1, 1, 1)
        # 通道注意力权重
        out = out.expand_as(x)
        # 与输入张量相乘
        return x * out


class ResidualBlock(nn.Module):
    """
    ResNet的基本残差块
    用于ResNet-18和ResNet-34
    """
    expansion = 1 # 基本残差块的输出通道数与输入通道数相同

    def __init__(self, in_channels, out_channels, stride=1):
        """
        初始化基本残差块
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 步长,默认为1
            downsample: 下采样模块,用于匹配通道数和尺寸,默认为None
        """
        super(ResidualBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 通道注意力模块
        # self.attn = ChannelAttention(out_channels)
        # 下采样模块
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        # 激活函数
        self.relu = nn.ReLU(inplace=True) # 参数表示是否在原地操作,True表示在原内存上操作,False表示在新内存上操作

    def forward(self, x):
        # 残差路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(x)
        # out = self.attn(out)
        out = self.relu(out)

        return out

class ResEmotionNet(nn.Module):
    """
    ResNet基础类,可用于构建不同深度的ResNet网络
    """
    def __init__(self, num_classes=7, zero_init_residual=False, dropout_rate=0.5):
        """
        初始化ResNet网络
        Args:
            block: 残差块类型(BasicBlock或Bottleneck)
            layers: 每个阶段的残差块数量
            num_classes: 分类类别数,FER-2013数据集有7个情绪类别
            zero_init_residual: 是否将残差块的最后一个BN层初始化为0
        """
        super(ResEmotionNet, self).__init__()
        self.in_channels = 64
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64) 
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.se = SEBlock(256)
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # 自适应平均池化层,将特征图的尺寸压缩到1x1

        # 构建残差层
        self.res_block1 = ResidualBlock(256, 512, stride=2)
        self.res_block2 = ResidualBlock(512, 1024, stride=2)
        self.res_block3 = ResidualBlock(1024, 2048, stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 全局平均池化、Dropout和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 自适应平均池化层,将特征图的尺寸压缩到1x1
        self.dropout1 = nn.Dropout(0.2)  # 添加Dropout层减少过拟合
        self.dropout2 = nn.Dropout(0.5)  # 添加Dropout层减少过拟合
        self.fc1 = nn.Linear(2048, 1024)  
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)
            
    
    def forward(self, x):

        # 输入x [batch_size, 3, 64, 64]，通过初始卷积层
        x = self.conv1(x) # 【batch_size, 64, 64, 64】
        x = self.bn1(x) # 【batch_size, 64, 64, 64】
        x = self.relu(x) # 【batch_size, 64, 64, 64】
        x = self.maxpool(x) # 【batch_size, 64, 32, 32】
        x = self.dropout1(x) # 【batch_size, 64, 32, 32】

        x = self.conv2(x) # 【batch_size, 128, 32, 32】
        x = self.bn2(x) # 【batch_size, 128, 32, 32】
        x = self.relu(x) # 【batch_size, 128, 32, 32】
        x = self.maxpool(x) # 【batch_size, 128, 16, 16】
        x = self.dropout1(x) # 【batch_size, 128, 16, 16】

        x = self.conv3(x) # 【batch_size, 256, 16, 16】
        x = self.bn3(x) # 【batch_size, 256, 16, 16】
        x = self.relu(x) # 【batch_size, 256, 16, 16】
        x = self.maxpool(x) # 【batch_size, 256, 8, 8】
        x = self.se(x) # 【batch_size, 256, 8, 8】

        x = self.res_block1(x) # 【batch_size, 512, 4, 4】
        x = self.res_block2(x) # 【batch_size, 1024, 2, 2】
        x = self.res_block3(x) # 【batch_size, 2048, 1, 1】
        # x = self.layer4(x) # 【batch_size, 2048, 1, 1】 

        x = self.pool(x) # 【batch_size, 2048, 1, 1】
        x = x.view(x.size(0), -1) # 【batch_size, 2048】
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout2(x)
        x = self.fc4(x) # 【batch_size, 7】

        return x
