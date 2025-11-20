import os
import torch
import yaml
from src.train.train_resnet import get_fer2013_dataloaders

# 简单测试脚本，用于验证数据集加载是否正常
def test_data_loading():
    # 加载配置
    config_path = 'src/configs/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 更新配置以适应FER-2013数据集
    config['data']['dataset_name'] = 'FER-2013'
    config['data']['root_dir'] = 'data/FER_2013'  # 使用实际的数据集路径
    config['data']['image_size'] = 48  # 数据集实际图像尺寸为48x48
    config['data']['batch_size'] = 8  # 使用小批量进行测试
    config['data']['num_workers'] = 2
    
    print("正在加载数据集...")
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_fer2013_dataloaders(config)
    
    # 直接获取数据集大小而不是通过批次计算
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)
    
    print(f"训练集大小: {train_size}")
    print(f"验证集大小: {val_size}")
    print(f"测试集大小: {test_size}")
    
    # 测试训练集的第一个批次
    print("\n测试训练集数据:")
    for images, labels in train_loader:
        print(f"图像批次形状: {images.shape}")
        print(f"标签批次形状: {labels.shape}")
        print(f"标签示例: {labels[:5].tolist()}") # 显示前5个标签
        # 定义FER-2013数据集的情绪标签映射
        emotion_map = {0: '愤怒', 1: '厌恶', 2: '恐惧', 3: '开心', 4: '难过', 5: '惊讶', 6: '中性'}
        print(f"标签映射: {[emotion_map[label.item()] for label in labels[:5]]}") # 显示标签映射
        print(f"图像数据类型: {images.dtype}")
        print(f"标签数据类型: {labels.dtype}")
        break  # 只测试第一个批次
    
    # 测试验证集的第一个批次
    print("\n测试验证集数据:")
    for images, labels in val_loader:
        print(f"图像批次形状: {images.shape}")
        print(f"标签批次形状: {labels.shape}")
        break
    
    # 测试测试集的第一个批次
    print("\n测试测试集数据:")
    for images, labels in test_loader:
        print(f"图像批次形状: {images.shape}")
        print(f"标签批次形状: {labels.shape}")
        break
    
    print("\n数据加载测试成功完成！")

if __name__ == '__main__':
    test_data_loading()