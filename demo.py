"""
演示脚本：测试驾驶员分心检测系统

这个脚本提供了一个简单的演示界面，展示如何使用我们的驾驶员分心检测模型。
即使没有真实数据集，也可以使用虚拟数据进行演示。
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 设置中文字体支持 - 添加更全面的备选字体列表以确保更好的中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Arial Unicode MS', 'DejaVu Sans',
                                   'Noto Sans CJK SC', 'Noto Sans CJK', 'Microsoft YaHei', 'SimSun',
                                   'KaiTi', 'FangSong', 'STSong', 'Malgun Gothic', 'Apple SD Gothic Neo']  # 用来正常显示中文标签
# 启用字体回退机制，当指定字体不包含某字符时尝试使用其他可用字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

print("=== 驾驶员分心检测系统演示 ===")
print("本演示将展示如何使用三阶段Mamba模型进行驾驶员分心检测")

# 检查项目结构
print("\n[1] 检查项目结构...")
required_dirs = ['src', 'src/configs', 'src/data', 'src/models', 'src/train', 'src/utils', 'results', 'logs']
for dir_path in required_dirs:
    if os.path.exists(dir_path):
        print(f"✓ 找到目录: {dir_path}")
    else:
        print(f"✗ 未找到目录: {dir_path}")

# 检查依赖
print("\n[2] 检查主要依赖...")
deps = ['torch', 'torchvision', 'numpy', 'matplotlib', 'mamba_ssm', 'tqdm', 'yaml']
for dep in deps:
    try:
        __import__(dep)
        print(f"✓ 已安装: {dep}")
    except ImportError:
        print(f"✗ 未安装: {dep}")

# 导入自定义模块
print("\n[3] 导入项目模块...")
try:
    from src.utils.utils import load_config, count_parameters, get_model_size
    from src.models.mamba_model import create_mamba_model, SimpleMambaDriverDistraction
    from src.data.dataset import DummyDriverDataset
    print("✓ 成功导入项目模块")
except Exception as e:
    print(f"✗ 导入模块失败: {e}")
    # 备用错误处理
    try:
        # 尝试单独导入各个模块
        from src.utils.utils import load_config
        print("✓ 成功导入配置函数")
    except:
        print("✗ 无法导入配置函数")

# 创建虚拟数据集演示
print("\n[4] 创建虚拟数据集演示...")
try:
    # 创建一个小的虚拟数据集
    dummy_dataset = DummyDriverDataset(num_samples=10, image_size=224)
    print(f"✓ 成功创建虚拟数据集，包含 {len(dummy_dataset)} 个样本")
    print(f"  - 类别数量: {dummy_dataset.num_classes}")
    print(f"  - 图像大小: {dummy_dataset.image_size}x{dummy_dataset.image_size}")
    
    # 显示前3个样本
    plt.figure(figsize=(15, 5))
    for i in range(min(3, len(dummy_dataset))):
        image, label = dummy_dataset[i]
        plt.subplot(1, 3, i+1)
        # 转换为numpy格式并显示
        img_np = image.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        plt.imshow(img_np)
        class_name = dummy_dataset.class_names.get(f'c{label}', f'类别{label}')
        plt.title(f"标签: {label} ({class_name})")
        plt.axis('off')
    plt.suptitle("虚拟数据集样本展示")
    plt.tight_layout()
    
    # 保存示例图像
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/dummy_samples.png')
    print("✓ 已保存示例图像到 results/dummy_samples.png")
except Exception as e:
    print(f"✗ 创建虚拟数据集失败: {e}")

# 模型架构演示
print("\n[5] 模型架构演示...")
try:
    # 检查设备是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建一个简单的Mamba模型实例
    model = SimpleMambaDriverDistraction(
        num_classes=10,
        img_size=224,
        embed_dim=64,  # 使用较小的维度以便演示
        depths=[1, 1, 1],  # 简化的三阶段
        ssm_rank=32
    )
    
    # 计算模型参数
    param_count = count_parameters(model)
    print(f"✓ 成功创建Mamba模型")
    print(f"  - 参数数量: {param_count:,}")
    print(f"  - 模型架构: 三阶段Mamba")
    print(f"  - 类别数量: 10")
    
    # 将模型移动到设备上
    model = model.to(device)
    
    # 测试模型前向传播
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"✓ 模型前向传播测试成功")
    print(f"  - 输入形状: {dummy_input.shape}")
    print(f"  - 输出形状: {output.shape}")
    
    # 显示模型结构概要
    print("\n模型结构概要:")
    print("1. 初始卷积特征提取层")
    print("2. 中间特征处理层")
    print("3. 三阶段Mamba序列建模层")
    print("4. 全局池化和分类层")
except Exception as e:
    print(f"✗ 模型演示失败: {e}")

# 训练流程说明
print("\n[6] 训练流程说明...")
print("训练过程将包括以下步骤:")
print("  1. 加载数据集（真实或虚拟）")
print("  2. 初始化三阶段Mamba模型")
print("  3. 设置优化器和学习率调度器")
print("  4. 多轮训练，每轮包括:")
print("     - 在训练集上训练模型")
print("     - 在验证集上评估性能")
print("     - 保存最佳模型权重")
print("  5. 生成训练历史图表")
print("  6. 在测试集上进行最终评估")

# 推理演示说明
print("\n[7] 推理功能说明...")
print("训练完成后，可以使用以下方式进行推理:")
print("  1. 单个图像预测")
print("  2. 批量图像预测")
print("  3. 预测结果可视化")
print("  4. 获取多类别概率分布")

# 使用指南
print("\n[8] 使用指南")
print("\n如何运行训练:")
print("  $ python src/train/train.py")
print("\n如何进行推理:")
print("  $ python src/utils/inference.py")
print("\n如何自定义配置:")
print("  编辑 src/configs/config.yaml 文件修改参数")

# 注意事项
print("\n[9] 注意事项")
print("  - 如果没有GPU，将自动使用CPU进行训练（但会比较慢）")
print("  - 首次运行会自动使用虚拟数据集进行演示")
print("  - 训练结果将保存在 results 和 checkpoints 目录")
print("  - 详细日志请查看 logs/training.log")

print("\n=== 演示完成 ===")
print("现在可以运行训练脚本开始训练模型！")