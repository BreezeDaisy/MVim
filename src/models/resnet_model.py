import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    ResNet的基本残差块
    用于ResNet-18和ResNet-34
    """
    expansion = 1 # 基本残差块的输出通道数与输入通道数相同

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        初始化基本残差块
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 步长,默认为1
            downsample: 下采样模块,用于匹配通道数和尺寸,默认为None
        """
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 下采样模块
        self.downsample = downsample
        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        # 残差路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 快捷路径
        if self.downsample is not None:
            identity = self.downsample(x) # 下采样模块，用于匹配通道数和尺寸
        # 残差连接
        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """
    ResNet基础类,可用于构建不同深度的ResNet网络
    """
    def __init__(self, block, layers, num_classes=7, zero_init_residual=False, dropout_rate=0.5):
        """
        初始化ResNet网络
        Args:
            block: 残差块类型(BasicBlock或Bottleneck)
            layers: 每个阶段的残差块数量
            num_classes: 分类类别数,FER-2013数据集有7个情绪类别
            zero_init_residual: 是否将残差块的最后一个BN层初始化为0
        """
        super(ResNet, self).__init__()
        self.in_channels = 64
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64) 
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        
        # 构建残差层
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 全局平均池化、Dropout和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 自适应平均池化层,将特征图的尺寸压缩到1x1
        self.dropout = nn.Dropout(dropout_rate)  # 添加Dropout层减少过拟合
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d): # 卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): # 批量归一化层
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 零初始化残差块的最后一个BN层
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        创建残差层
        Args:
            block: 残差块类型
            out_channels: 输出通道数
            blocks: 残差块数量
            stride: 步长
        Returns:
            残差层
        """
        downsample = None
        # 如果步长不为1或输入通道数不等于输出通道数*expansion，则需要下采样
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * block.expansion, 
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        # 添加第一个残差块
        layers.append(block(
            self.in_channels, out_channels, stride, downsample
        ))
        # 更新输入通道数
        self.in_channels = out_channels * block.expansion
        # 添加剩余的残差块
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):

        # 输入x [3, 48, 48]，通过初始卷积层
        x = self.conv1(x) # 【64, 24, 24】
        x = self.bn1(x) # 【64, 24, 24】
        x = self.relu(x) # 【64, 24, 24】
        x = self.maxpool(x) # 【64, 12, 12】

        x = self.layer1(x) # 【64, 12, 12】
        x = self.layer2(x) # 【128, 6, 6】
        x = self.layer3(x) # 【256, 3, 3】
        x = self.layer4(x) # 【512, 1, 1】 向下取整

        x = self.avgpool(x) # 【512, 1, 1】
        x = torch.flatten(x, 1) # 【512】
        x = self.dropout(x)  # 在全连接层前应用Dropout
        x = self.fc(x) # 【7】

        return x

def resnet18(num_classes=7, zero_init_residual=False, dropout_rate=0.5):
    """
    创建ResNet-18模型
    Args:
        num_classes: 分类类别数,FER-2013数据集有7个情绪类别
        zero_init_residual: 是否将残差块的最后一个BN层初始化为0
    Returns:
        ResNet-18模型
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, zero_init_residual, dropout_rate)

def resnet34(num_classes=7, zero_init_residual=False, dropout_rate=0.5):
    """
    创建ResNet-34模型
    Args:
        num_classes: 分类类别数
        zero_init_residual: 是否将残差块的最后一个BN层初始化为0
    Returns:
        ResNet-34模型
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, zero_init_residual, dropout_rate)

def create_resnet_model(config):
    """
    根据配置创建ResNet模型
    Args:
        config: 配置字典，包含模型参数
    Returns:
        ResNet模型
    """
    # 从配置中获取参数
    model_depth = config['model'].get('depth')
    num_classes = config['model'].get('num_classes')
    zero_init_residual = config['model'].get('zero_init_residual', False)
    dropout_rate = config['model'].get('dropout_rate', 0.5)
    
    # 根据深度创建相应的模型
    if model_depth == '18':
        model = resnet18(num_classes, zero_init_residual, dropout_rate)
    elif model_depth == '34':
        model = resnet34(num_classes, zero_init_residual, dropout_rate)
    else:
        raise ValueError(f"不支持的ResNet深度: {model_depth}")
    
    # 如果配置中指定了预训练模型，则加载
    if 'model_checkpoints' in config['model'] and config['model']['model_checkpoints']:
        checkpoint_path = config['model']['model_checkpoints']
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cuda') # 加载到GPU
            # 如果是完整的模型字典，直接加载
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 否则尝试直接加载
                model.load_state_dict(checkpoint)
            print(f"成功加载预训练模型: {checkpoint_path}")
        except Exception as e:
            print(f"加载预训练模型失败: {e}")
    
    return model