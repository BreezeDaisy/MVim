import os
import yaml
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from datetime import datetime

def set_seed(seed=42):
    """设置随机种子，确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, save_path):
    """保存配置文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def create_directory(path):
    """创建目录，如果已存在则不报错"""
    os.makedirs(path, exist_ok=True)

def get_timestamp():
    """获取当前时间戳"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def plot_images(images, labels, class_names, nrows=3, ncols=3, figsize=(12, 12)):
    """绘制图像和标签"""
    plt.figure(figsize=figsize)
    for i, (img, label) in enumerate(zip(images[:nrows*ncols], labels[:nrows*ncols])):
        plt.subplot(nrows, ncols, i+1)
        # 如果图像是张量，需要转换为numpy格式
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
            # 反归一化
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f"Label: {class_names[f'c{label}']}")
        plt.axis('off')
    plt.tight_layout()
    return plt

def visualize_prediction(image, true_label, pred_label, confidence, class_names):
    """可视化单个预测结果"""
    plt.figure(figsize=(8, 6))
    
    # 如果图像是张量，需要转换为numpy格式
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
    
    plt.imshow(image)
    plt.title(f"True: {class_names[f'c{true_label}']}\nPred: {class_names[f'c{pred_label}']} ({confidence:.2%})")
    plt.axis('off')
    return plt

def calculate_class_weights(labels, num_classes):
    """计算类别权重，用于不平衡数据集"""
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts)
    return torch.tensor(class_weights, dtype=torch.float32)

def log_metrics(metrics, log_file):
    """记录评估指标到文件"""
    with open(log_file, 'a') as f:
        f.write(f"{get_timestamp()}\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("-" * 50 + "\n")

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(model):
    """获取模型大小（MB）"""
    torch.save(model.state_dict(), 'temp_model.pth')
    size = os.path.getsize('temp_model.pth') / (1024 * 1024)
    os.remove('temp_model.pth')
    return size

def preprocess_image(image_path, image_size=224):
    """预处理单张图像用于推理"""
    from torchvision import transforms
    
    # 定义预处理变换
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度
    
    return image_tensor

def postprocess_prediction(output, class_names):
    """后处理模型输出"""
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = f'c{predicted_idx.item()}'
    class_name = class_names.get(predicted_class, 'Unknown')
    
    return {
        'class_index': predicted_idx.item(),
        'class': predicted_class,
        'class_name': class_name,
        'confidence': confidence.item(),
        'probabilities': probabilities.squeeze().tolist()
    }

def save_inference_results(results, save_path):
    """保存推理结果"""
    import json
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def load_image_from_bytes(image_bytes):
    """从字节数据加载图像"""
    from io import BytesIO
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    return image

def plot_confidence_distribution(probabilities, class_names, figsize=(10, 6)):
    """绘制置信度分布"""
    plt.figure(figsize=figsize)
    class_indices = list(range(len(class_names)))
    class_labels = [class_names[f'c{i}'] for i in class_indices]
    plt.bar(class_labels, probabilities)
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Class Probability Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return plt

def analyze_gradients(model):
    """分析模型梯度（用于调试梯度消失/爆炸问题）"""
    grad_norms = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append((name, grad_norm))
    
    # 按梯度范数排序
    grad_norms.sort(key=lambda x: x[1], reverse=True)
    
    return grad_norms

def freeze_layers(model, freeze_until=None):
    """冻结模型层"""
    layers = list(model.named_parameters())
    
    if freeze_until is not None:
        # 冻结到指定层
        for name, param in layers[:freeze_until]:
            param.requires_grad = False
    else:
        # 冻结所有层
        for name, param in model.named_parameters():
            param.requires_grad = False
    
    return model

def unfreeze_layers(model):
    """解冻模型层"""
    for name, param in model.named_parameters():
        param.requires_grad = True
    
    return model

def export_onnx(model, dummy_input, output_path, opset_version=11):
    """导出模型为ONNX格式"""
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"模型已导出为ONNX格式: {output_path}")

def print_model_summary(model, input_size=(3, 224, 224)):
    """打印模型结构摘要"""
    from torchsummary import summary
    summary(model, input_size=input_size)