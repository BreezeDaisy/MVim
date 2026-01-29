import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from Face_Pose import face_network
import matplotlib.pyplot as plt
import numpy as np

# ======================================================
# 配置日志
# ======================================================

def setup_logger():
    """
    配置日志记录器
    """
    # 创建日志目录
    log_dir = "Face_Pose/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志记录器
    logger = logging.getLogger("train_face")
    logger.setLevel(logging.INFO)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 文件处理器，将日志保存到文件中
        file_handler = logging.FileHandler(os.path.join(log_dir, "train_face.log"))
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器，将日志输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器，定义日志格式。时间戳-日志记录器名称-日志级别-日志消息
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# 初始化日志记录器
logger = setup_logger()

# ======================================================
# 配置参数
# ======================================================

# 数据集路径
DATASET_DIR = "Face_Pose/Constructed_Emotion_Dataset"

# 模型保存路径
MODEL_SAVE_DIR = "Face_Pose/models/face"

# 可视化保存路径
VISUAL_SAVE_DIR = "Face_Pose/results/260129/face/visualizations"

# 情绪类别映射
EMOTION_LABELS = {
    "antipathic": 0,
    "fear": 1,
    "happy": 2,
    "neutral": 3,
    "sad": 4,
    "surprise": 5
}

# 数据变换
# TRANSFORMS = transforms.Compose([
#     transforms.Resize((112, 112)),  # 调整为模型输入大小，不剪裁
#     transforms.ToTensor(),          # 转换为张量
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],  # ImageNet均值
#         std=[0.229, 0.224, 0.225]    # ImageNet标准差
#     )
# ])
TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(112, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ======================================================
# 数据集类
# ======================================================

class EmotionDataset(Dataset):
    def __init__(self, dataset_dir, split="train", transform=None):
        """
        情绪识别数据集
        Args:
            dataset_dir: 数据集根目录
            split: 数据集划分，"train" 或 "val"
            transform: 数据预处理变换
        """
        self.dataset_dir = os.path.join(dataset_dir, split)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        for emotion, label in EMOTION_LABELS.items():
            emotion_dir = os.path.join(self.dataset_dir, emotion)
            if not os.path.exists(emotion_dir):
                continue
            
            for filename in os.listdir(emotion_dir):
                if filename.endswith(".jpg"):
                    img_path = os.path.join(emotion_dir, filename)
                    self.image_paths.append(img_path)
                    self.labels.append(label) # 将标签映射为0-5的整数
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        image = Image.open(img_path).convert("RGB")
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ======================================================
# 训练函数
# ======================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个epoch
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
    Returns:
        float: 平均损失
        float: 准确率
    """
    model.train() # 切换到训练模式
    running_loss = 0.0 # 累计损失
    correct = 0 # 累计正确预测数
    total = 0 # 累计样本数
    
    with tqdm(dataloader, desc="training") as pbar: # 进度条
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels) # 计算损失          
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            # 更新进度条
            pbar.set_postfix({
                "loss": running_loss / (pbar.n + 1),
                "acc": 100. * correct / total
            })
    
    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device, return_confusion=False):
    """
    验证模型
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        return_confusion: 是否返回混淆矩阵
    Returns:
        float: 平均损失
        float: 准确率
        np.ndarray: 混淆矩阵(如果return_confusion为True)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 初始化混淆矩阵
    if return_confusion:
        num_classes = 6  # 情绪类别数
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        with tqdm(dataloader, desc="validating") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)               
                # 统计
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()               
                # 更新混淆矩阵
                if return_confusion:
                    for label, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                        confusion_matrix[label, pred] += 1                
                # 更新进度条
                pbar.set_postfix({
                    "loss": running_loss / (pbar.n + 1),
                    "acc": 100. * correct / total
                })
    if return_confusion:
        return running_loss / len(dataloader), 100. * correct / total, confusion_matrix
    else:
        return running_loss / len(dataloader), 100. * correct / total

def visualize_training(train_loss, train_acc, val_loss, val_acc):
    """
    可视化训练过程
    Args:
        train_loss: 训练损失历史
        train_acc: 训练准确率历史
        val_loss: 验证损失历史
        val_acc: 验证准确率历史
    """
    # 创建保存目录
    visual_dir = VISUAL_SAVE_DIR
    os.makedirs(visual_dir, exist_ok=True)
    # 创建画布和子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # 绘制损失曲线
    ax1.plot(range(1, len(train_loss) + 1), train_loss, label="train_loss", color="blue")
    ax1.plot(range(1, len(val_loss) + 1), val_loss, label="val_loss", color="red")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss Curves")
    ax1.legend()
    ax1.grid(True)
    # 绘制准确率曲线
    ax2.plot(range(1, len(train_acc) + 1), train_acc, label="train_acc", color="blue")
    ax2.plot(range(1, len(val_acc) + 1), val_acc, label="val_acc", color="red")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training and Validation Accuracy Curves")
    ax2.legend()
    ax2.grid(True)
    # 调整布局
    plt.tight_layout()
    # 保存图表
    save_path = os.path.join(visual_dir, "training_curves.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    logger.info(f"训练曲线已保存到: {save_path}")


def visualize_confusion_matrix(confusion_matrix, class_names):
    """
    可视化混淆矩阵
    Args:
        confusion_matrix: 混淆矩阵
        class_names: 类别名称列表
    """
    # 创建保存目录
    visual_dir = VISUAL_SAVE_DIR
    os.makedirs(visual_dir, exist_ok=True)
    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    # 左侧：原始混淆矩阵（整数）
    im1 = ax1.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.figure.colorbar(im1, ax=ax1)
    ax1.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix (Raw Counts)',
           xlabel='Predicted Class',
           ylabel='True Class')
    # 旋转x轴标签
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # 在矩阵中添加文本
    fmt = 'd'
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax1.text(j, i, format(confusion_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    # 右侧：归一化混淆矩阵（0-1）
    confusion_matrix_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    im2 = ax2.imshow(confusion_matrix_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.figure.colorbar(im2, ax=ax2)
    ax2.set(xticks=np.arange(confusion_matrix_norm.shape[1]),
           yticks=np.arange(confusion_matrix_norm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix (Normalized)',
           xlabel='Predicted Class',
           ylabel='True Class')
    # 旋转x轴标签
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # 在矩阵中添加文本
    fmt = '.2f'
    thresh = confusion_matrix_norm.max() / 2.
    for i in range(confusion_matrix_norm.shape[0]):
        for j in range(confusion_matrix_norm.shape[1]):
            ax2.text(j, i, format(confusion_matrix_norm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion_matrix_norm[i, j] > thresh else "black")
    # 保存图像
    plt.tight_layout()
    save_path = os.path.join(visual_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    logger.info(f"混淆矩阵已保存到: {save_path}")
    plt.close()

# ======================================================
# 主函数
# ======================================================

def main(args):
    # 检查设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # 检查是否有多个GPU
        if torch.cuda.device_count() > 1:
            logger.info(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
    else:
        device = torch.device("cpu")
        logger.info("使用CPU进行训练")
    # 创建模型保存目录
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    # 加载数据集
    logger.info("加载数据集...")
    train_dataset = EmotionDataset(DATASET_DIR, split="train", transform=TRANSFORMS)
    val_dataset = EmotionDataset(DATASET_DIR, split="val", transform=TRANSFORMS)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    # 初始化模型
    logger.info("初始化模型...")
    model = face_network.ImageClassifier() # 初始化模型
    # 使用多个GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    # 定义损失函数和优化器，使用交叉熵损失和Adam优化器
    criterion = nn.CrossEntropyLoss() 
    # 换成AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 学习率调度器，每10个epoch学习率衰减为原来的0.1倍
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2) # 每5个epoch学习率衰减为原来的0.2倍
    # 训练循环
    best_val_acc = 0.0
    # 用于可视化的历史数据
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    logger.info("开始训练...")
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.info("-" * 50)
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        logger.info(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")       
        # 记录历史数据
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)        
        # 学习率调度
        scheduler.step()       
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"保存最佳模型到: {model_path}")   
    # 保存最终模型
    logger.info(f"\n训练完成!")
    logger.info(f"保存最佳模型到: {model_path}")
    logger.info(f"最佳验证准确率: {best_val_acc:.2f}%")
    # 可视化训练过程
    visualize_training(train_loss_history, train_acc_history, val_loss_history, val_acc_history)
    # 计算并可视化混淆矩阵
    logger.info("计算混淆矩阵...")
    _, _, confusion_matrix = validate(model, val_loader, criterion, device, return_confusion=True)
    # 定义类别名称
    class_names = ['Antipathic', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    # 可视化混淆矩阵
    visualize_confusion_matrix(confusion_matrix, class_names)

# ======================================================
# 入口点
# ======================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="批次大小"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="训练轮数"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0002,
        help="学习率"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="权重衰减"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="数据加载器工作线程数"
    )
    
    args = parser.parse_args()
    main(args)