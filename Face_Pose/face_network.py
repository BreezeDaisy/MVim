import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """
    残差块
    包含两个卷积层和一个跳跃连接
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 跳跃连接（处理通道数和步长不同的情况）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 跳跃连接
        out += self.shortcut(x)
        out = self.relu(out)
        
        return out



        
class ImageClassifier(nn.Module):
    """
    图像分类器网络
    输入: (batch_size, 3, input_size, input_size)
    输出: (batch_size, 6) - 6分类
    """
    def __init__(self, input_size=112):
        super().__init__()
        self.input_size = input_size
        
        # 卷积特征提取部分
        self.features = nn.Sequential(
            # 初始卷积: (3, input_size, input_size) → (32, input_size//2, input_size//2)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), # inplace表示直接在输入上操作，不占用额外内存
            # dropout层，防止过拟合
            nn.Dropout(p=0.3, inplace=False),

            # 残差块1: (32, input_size//2, input_size//2) → (64, input_size//4, input_size//4)
            ResidualBlock(in_channels=32, out_channels=64, stride=2),

            ChannelAttention(64), # 通道注意力机制

            # 池化: (64, input_size//4, input_size//4) → (64, (input_size//4 + 2)//3, (input_size//4 + 2)//3)
            nn.MaxPool2d(kernel_size=4, stride=3, padding=1),

            # 残差块2: 上一步结果 → (128, 上一步结果//2, 上一步结果//2)
            ResidualBlock(in_channels=64, out_channels=128, stride=2),

            # 池化: 上一步结果 → (128, 上一步结果//2, 上一步结果//2)
            nn.MaxPool2d(kernel_size=2, stride=2)               
        )
        
        # 计算分类器输入维度
        # 模拟前向传播计算特征图尺寸
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, input_size, input_size)
            features_output = self.features(dummy_input)
            classifier_input_dim = features_output.numel() // features_output.shape[0]
        
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Flatten(),
            
            # 全连接层1
            nn.Linear(classifier_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3, inplace=False),
            
            # 全连接层2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3, inplace=False),
            
            # 输出层
            nn.Linear(128, 6)
        )
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，形状为 (batch_size, 3, H, W)
        Returns:
            输出张量，形状为 (batch_size, 6)    
        """
        # 特征提取
        x = self.features(x)
        
        # 分类
        x = self.classifier(x)
        
        return x

# 验证网络结构和维度
if __name__ == "__main__":
    # 测试不同输入尺寸
    test_sizes = [112, 128, 96]
    
    for size in test_sizes:
        print(f"\n测试输入尺寸: {size}x{size}")
        # 创建模型实例
        model = ImageClassifier(input_size=size)
        # 生成测试输入：batch_size=4, 3通道
        test_input = torch.randn(4, 3, size, size)
        # 前向传播
        output = model(test_input)
        
        # 打印各层输出维度（验证是否符合要求）
        print("输入维度:", test_input.shape)
        print("最终输出维度:", output.shape)  # 应为(4,6)，对应4个样本+6分类
        print("模型总参数量:", sum(p.numel() for p in model.parameters()))
    
    # 测试默认输入尺寸
    print("\n测试默认输入尺寸: 112x112")
    model = ImageClassifier()  # 使用默认input_size=112
    test_input = torch.randn(4, 3, 112, 112)
    output = model(test_input)
    print("输入维度:", test_input.shape)
    print("最终输出维度:", output.shape)