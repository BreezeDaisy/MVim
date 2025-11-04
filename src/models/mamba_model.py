import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba  # 导入Mamba模型

class ChannelAttention(nn.Module):
    """
    通道注意力模块
    使用全局平均池化和最大池化来捕获通道维度上的信息
    in_channels:输入通道数量,由Mamba模型的输出通道数决定,来自 AttentionModule(dim) 的参数传递
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [batch_size, seq_len, in_channels]
        batch_size, seq_len, in_channels = x.size()
        
        # 平均池化路径
        avg_out = self.avg_pool(x.transpose(1, 2)).squeeze(-1)  # [batch_size, in_channels]
        avg_out = self.fc(avg_out).unsqueeze(2)  # [batch_size, in_channels, 1]
        
        # 最大池化路径
        max_out = self.max_pool(x.transpose(1, 2)).squeeze(-1)  # [batch_size, in_channels]
        max_out = self.fc(max_out).unsqueeze(2)  # [batch_size, in_channels, 1]
        
        # 拼接两条路径的输出
        out = avg_out + max_out
        out = out.transpose(1, 2)  # [batch_size, 1, in_channels]
        
        # 应用注意力权重
        return x * out.expand_as(x)

class SpatialAttention(nn.Module):
    """
    空间注意力模块
    捕获序列维度上的空间信息
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: [batch_size, seq_len, in_channels]
        batch_size, seq_len, in_channels = x.size()
        
        # 在通道维度上进行平均和最大池化
        avg_out = torch.mean(x, dim=2, keepdim=True)  # [batch_size, seq_len, 1]
        max_out, _ = torch.max(x, dim=2, keepdim=True)  # [batch_size, seq_len, 1]
        
        # 拼接池化结果
        out = torch.cat([avg_out, max_out], dim=2)  # [batch_size, seq_len, 2]
        out = out.transpose(1, 2)  # [batch_size, 2, seq_len]
        
        # 通过卷积层
        out = self.conv(out)  # [batch_size, 1, seq_len]
        out = self.sigmoid(out).transpose(1, 2)  # [batch_size, seq_len, 1]
        
        # 应用注意力权重
        return x * out.expand_as(x)

class AttentionModule(nn.Module):
    """
    组合通道注意力和空间注意力的模块
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # 先应用通道注意力，再应用空间注意力
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class MambaBlock(nn.Module):
    """
    Mamba块,包含Mamba层、归一化、注意力机制和残差连接
    """
    def __init__(self, dim, ssm_rank=64, dropout_rate=0.1, use_attention=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,           # 模型维度
            d_state=ssm_rank,      # SSM状态维度
            d_conv=4,              # 卷积核维度
            expand=2               # 扩展因子
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.use_attention = use_attention
        if use_attention:
            # 添加注意力模块
            self.attention = AttentionModule(dim)
    
    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        
        # 应用注意力机制
        if self.use_attention:
            x = self.attention(x)
            
        x = self.dropout(x)
        x = x + residual
        return x

class MambaStage(nn.Module):
    """
    Mamba阶段,包含多个Mamba块
    """
    def __init__(self, dim, depth, ssm_rank=64, dropout_rate=0.1, use_attention=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(dim, ssm_rank, dropout_rate, use_attention) for _ in range(depth)
        ])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class PatchEmbed(nn.Module):
    """
    图像分块嵌入
    将图像分割成多个patches并投影到高维空间
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: [batch_size, channels, height, width]
        batch_size = x.shape[0]
        x = self.proj(x)  # [batch_size, embed_dim, grid_size, grid_size]
        x = x.flatten(2)  # [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        return x

class MambaDriverDistraction(nn.Module):
    """
    三阶段Mamba架构用于驾驶员分心检测
    """
    def __init__(self,
                 num_classes=10,
                 img_size=224,
                 patch_size=16,
                 embed_dim=256,
                 depths=[2, 2, 2],  # 三阶段的深度,每个阶段的Mamba块数量
                 ssm_rank=64,
                 dropout_rate=0.1,
                 use_attention=True):
        super().__init__()
        
        # 图像分块嵌入
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim
        )
        
        # 位置嵌入
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # 三阶段Mamba
        self.stages = nn.ModuleList()
        current_dim = embed_dim
        
        for i, depth in enumerate(depths):
            # 每个阶段可能有不同的维度（可选）
            # 这里简化处理，保持维度一致
            stage = MambaStage(
                dim=current_dim,
                depth=depth,
                ssm_rank=ssm_rank,
                dropout_rate=dropout_rate,
                use_attention=use_attention
            )
            self.stages.append(stage)
            
            # 阶段间的过渡（可选）
            if i < len(depths) - 1:
                # 可以添加维度变换或池化操作
                pass
        
        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 对于线性层，使用截断正态分布初始化权重
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            # 对于卷积层，使用Kaiming正态分布初始化权重
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            # 对于层归一化层，使用常量初始化权重
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def forward_features(self, x):
        # x: [batch_size, 3, img_size, img_size]
        x = self.patch_embed(x)  # [batch_size, num_patches, embed_dim]
        
        # 添加cls_token
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, num_patches + 1, embed_dim]
        
        # 添加位置嵌入
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # 通过三阶段Mamba
        for stage in self.stages:
            x = stage(x)
        
        # 使用cls token的输出
        x = self.norm(x)
        return x[:, 0]  # [batch_size, embed_dim]
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)  # [batch_size, num_classes]
        return x

# 另一个版本：使用更简洁的Mamba架构
class SimpleMambaDriverDistraction(nn.Module):
    """
    简化版三阶段Mamba架构,更适合驾驶员分心检测任务
    """
    def __init__(self,
                 num_classes=10,
                 img_size=224,
                 embed_dim=256,
                 depths=[2, 2, 2],
                 ssm_rank=64,
                 dropout_rate=0.1,
                 use_attention=True):
        super().__init__()
        
        # 初始卷积特征提取
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, embed_dim // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 中间卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 计算序列长度（用于Mamba）
        self.seq_len = (img_size // 16) ** 2
        
        # 三阶段Mamba
        self.mamba_stages = nn.ModuleList()
        current_dim = embed_dim
        
        for depth in depths:
            stage_blocks = []
            for _ in range(depth):
                stage_blocks.append(MambaBlock(current_dim, ssm_rank, dropout_rate, use_attention))
            self.mamba_stages.append(nn.Sequential(*stage_blocks))
        
        # 分类头
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # x: [batch_size, 3, img_size, img_size]
        
        # 通过卷积层提取特征
        x = self.conv1(x)  # [batch_size, embed_dim//4, H/4, W/4]
        x = self.conv2(x)  # [batch_size, embed_dim//2, H/8, W/8]
        x = self.conv3(x)  # [batch_size, embed_dim, H/16, W/16]
        
        # 转换为序列格式 [batch_size, seq_len, embed_dim]
        batch_size = x.shape[0]
        x = x.flatten(2)  # [batch_size, embed_dim, H*W/256]
        x = x.transpose(1, 2)  # [batch_size, seq_len, embed_dim]
        
        # 通过三阶段Mamba
        for stage in self.mamba_stages:
            x = stage(x)
        
        # 全局平均池化
        x = x.transpose(1, 2)  # [batch_size, embed_dim, seq_len]
        x = self.global_pool(x).squeeze(-1)  # [batch_size, embed_dim]
        
        # 分类
        x = self.dropout(x)
        x = self.fc(x)  # [batch_size, num_classes]
        
        return x

# 模型工厂函数
def create_mamba_model(config):
    """
    根据配置创建Mamba模型
    """
    model_type = config['model'].get('architecture', 'simple')
    # 获取是否使用注意力机制的配置，默认为True
    use_attention = config['model'].get('use_attention', True)
    
    if model_type == 'simple':
        model = SimpleMambaDriverDistraction(
            num_classes=config['model']['num_classes'],
            img_size=config['data']['image_size'],
            embed_dim=config['model']['embed_dim'],
            depths=config['model']['depths'],
            ssm_rank=config['model']['ssm_rank'],
            dropout_rate=config['model']['dropout_rate'],
            use_attention=use_attention
        )
    elif model_type == 'mamba':
        model = MambaDriverDistraction(
            num_classes=config['model']['num_classes'],
            img_size=config['data']['image_size'],
            embed_dim=config['model']['embed_dim'],
            depths=config['model']['depths'],
            ssm_rank=config['model']['ssm_rank'],
            dropout_rate=config['model']['dropout_rate'],
            use_attention=use_attention
        )
    else :
        raise ValueError(f"未知模型类型: {model_type}")
    
    return model