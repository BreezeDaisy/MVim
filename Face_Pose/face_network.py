import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    位置嵌入模块
    为输入特征添加位置信息
    使用向量化操作，提高计算效率
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
    
    def forward(self, x):
        b, c, h, w = x.shape
        device = x.device
        
        # 计算位置编码的频率
        div_term = torch.exp(torch.arange(0, self.channels, 2, device=device) * 
                           (-torch.log(torch.tensor(10000.0, device=device)) / self.channels))
        
        # 创建位置索引
        position_h = torch.arange(0, h, device=device).unsqueeze(1).repeat(1, w)
        position_w = torch.arange(0, w, device=device).unsqueeze(0).repeat(h, 1)
        
        # 计算正弦和余弦编码
        pe = torch.zeros(1, self.channels, h, w, device=device)
        
        # 高度方向的位置编码
        pe[0, 0::2, :, :] = torch.sin(position_h.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        pe[0, 1::2, :, :] = torch.cos(position_h.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        
        # 宽度方向的位置编码（添加到现有编码上）
        if self.channels > 2:
            # 计算宽度方向编码的通道数
            width_channels = self.channels // 4
            if width_channels > 0:
                pe[0, 2::4, :, :] += torch.sin(position_w.unsqueeze(0) * div_term[:width_channels].unsqueeze(1).unsqueeze(2))
                pe[0, 3::4, :, :] += torch.cos(position_w.unsqueeze(0) * div_term[:width_channels].unsqueeze(1).unsqueeze(2))
        
        # 调整通道数以匹配输入
        pe = pe[:, :c, :, :].repeat(b, 1, 1, 1)
        
        return x + pe

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

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
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


class EnhancedBlock(nn.Module):
    """
    增强型网络块
    结构：下采样 → 归一化 → SE激励层 → 卷积FFN层
    归一化层接受下采样后和下采样前的相加作为输入
    最后输出为归一化之前和卷积FFN后相加的结果
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # 下采样
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                      stride=stride, padding=1, bias=False),
        )
        
        # 归一化
        self.bn = nn.BatchNorm2d(out_channels)
        
        # SE激励层
        self.se = SEBlock(out_channels)
        
        # 卷积FFN层
        self.ffn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 跳跃连接（处理通道数和步长不同的情况）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
            )

    def forward(self, x):
        # 下采样
        downsampled = self.downsample(x)
        
        # 跳跃连接
        shortcut = self.shortcut(x)
        
        # 归一化层接受下采样后和下采样前的相加作为输入
        norm_input = downsampled + shortcut
        normed = self.bn(norm_input)
        
        # SE激励层
        se_out = self.se(normed)
        
        # 卷积FFN层
        ffn_out = self.ffn(se_out)
        
        # 最后输出为归一化之前和卷积FFN后相加的结果
        out = norm_input + ffn_out
        out = F.relu(out, inplace=True)
        
        return out



        
class ImageClassifier(nn.Module):
    """
    图像分类器网络
    输入: (batch_size, 3, input_size, input_size)
    输出: (batch_size, 6) - 6分类
    """
    def __init__(self, input_size=224):
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

            # 增强型块1: (32, input_size//2, input_size//2) → (64, input_size//4, input_size//4)
            EnhancedBlock(in_channels=32, out_channels=64, stride=2),

            # 池化: (64, input_size//4, input_size//4) → (64, (input_size//4 + 2)//3, (input_size//4 + 2)//3)
            nn.MaxPool2d(kernel_size=4, stride=3, padding=1),

            # 增强型块2: 上一步结果 → (128, 上一步结果//2, 上一步结果//2)
            EnhancedBlock(in_channels=64, out_channels=128, stride=2),

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
    test_sizes = [112, 128, 224]
    
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