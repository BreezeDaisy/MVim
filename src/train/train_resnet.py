import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import yaml
import logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
from datetime import datetime

# 导入自定义模块
from src.models.resnet_model import create_resnet_model
from src.utils.utils import load_config, setup_matplotlib

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/training_resnet.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FER2013Dataset(torch.utils.data.Dataset):
    """
    FER-2013数据集的自定义数据集类
    用于从标准数据集格式加载和预处理数据
    """
    def __init__(self, csv_file, transform=None):
        """
        初始化数据集
        
        Args:
            csv_file: CSV文件路径，包含图像数据和标签
            transform: 应用于图像的转换
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        # FER-2013数据集的情绪标签映射
        self.emotion_map = {
            0: '愤怒',
            1: '厌恶',
            2: '恐惧',
            3: '开心',
            4: '难过',
            5: '惊讶',
            6: '中性'
        }
    
    def __len__(self):
        """
        返回数据集长度
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            图像和标签的元组
        """
        # 获取情绪标签
        emotion = self.data.iloc[idx, 0]
        # 获取像素数据并重塑为图像
        pixels = self.data.iloc[idx, 1].split(' ')
        image = np.array(pixels, dtype=np.uint8).reshape(48, 48)
        # 扩展为RGB格式
        image = np.stack([image, image, image], axis=0)
        # 转换为张量
        image = torch.tensor(image, dtype=torch.float32)
        # 应用变换
        if self.transform:
            image = self.transform(image)
        return image, emotion

class FER2013FolderDataset(torchvision.datasets.ImageFolder):
    """
    基于文件夹结构的FER-2013数据集类
    使用ImageFolder加载按情绪分类的图像
    """
    def __init__(self, root, transform=None, target_transform=None):
        """
        初始化数据集
        
        Args:
            root: 数据集根目录路径
            transform: 应用于图像的转换
            target_transform: 应用于标签的转换
        """
        super().__init__(root, transform=transform, target_transform=target_transform)
        # FER-2013数据集的情绪标签映射（按文件夹名称顺序）
        self.emotion_map = {
            0: 'angry',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'neutral',
            5: 'sad',
            6: 'surprise'
        }
        # 确保类别索引与情绪映射一致
        self.class_to_idx = {
            'angry': 0,
            'disgust': 1,
            'fear': 2,
            'happy': 3,
            'neutral': 4,
            'sad': 5,
            'surprise': 6
        }

def get_fer2013_dataloaders(config):
    """
    获取FER-2013数据集的数据加载器
    Args:
        config: 配置字典
    Returns:
        训练、验证和测试数据加载器的元组
    """
    # 定义训练集变换 - 增强版本以提高泛化能力
    train_transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.RandomResizedCrop(config['data']['image_size'], scale=(0.9, 1.0)),  # 随机裁剪
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1
        ),  # 更强烈的颜色变换
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),  # 高斯模糊
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 定义验证集和测试集变换（不包含数据增强）
    val_test_transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集（使用文件夹结构）
    train_dir = os.path.join(config['data']['root_dir'], 'train')
    test_dir = os.path.join(config['data']['root_dir'], 'test')
    
    # 创建数据集
    train_dataset = FER2013FolderDataset(train_dir, transform=train_transform)
    
    # 加载测试集并将其分为验证集和测试集（50%:50%）
    all_test_dataset = FER2013FolderDataset(test_dir, transform=val_test_transform)
    val_size = len(all_test_dataset) // 2
    test_size = len(all_test_dataset) - val_size
    val_dataset, test_dataset = random_split(
        all_test_dataset, [val_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

class ResNetTrainer:
    """
    ResNet模型训练器类
    用于训练和评估ResNet模型在FER-2013数据集上的性能
    """
    def __init__(self, config):
        """
        初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = create_resnet_model(config).to(self.device)
        
        # 创建损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 创建学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 准备数据加载器
        self.train_loader, self.val_loader, self.test_loader = get_fer2013_dataloaders(config)
        
        # 初始化训练记录
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_score = 0.0
        
        # 早停机制参数
        self.early_stopping_patience = self.config['train'].get('early_stopping_patience', 15)
        self.early_stopping_counter = 0
        self.early_stopping_enabled = self.early_stopping_patience > 0
        
        # 创建所有必要的保存目录
        for dir_path in ['checkpoint_dir', 'logs_dir', 'results_dir']:
            if dir_path in config['paths']:
                os.makedirs(config['paths'][dir_path], exist_ok=True)
    
    def _create_optimizer(self):
        """
        创建优化器
        
        Returns:
            优化器对象
        """
        if self.config['train']['optimizer'] == 'AdamW':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['train']['learning_rate'],
                weight_decay=self.config['train']['weight_decay']
            )
        elif self.config['train']['optimizer'] == 'Adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['train']['learning_rate'],
                weight_decay=self.config['train']['weight_decay']
            )
        else:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['train']['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['train']['weight_decay']
            )
        return optimizer
    
    def _create_scheduler(self):
        """
        创建学习率调度器
        
        Returns:
            学习率调度器对象
        """
        if self.config['train']['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['train']['epochs'],
                eta_min=0
            )
        elif self.config['train']['scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config['train']['scheduler'] == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=10,
                verbose=True
            )
        else:
            scheduler = None
        return scheduler
    
    def train_epoch(self):
        """
        训练一个epoch
        
        Returns:
            平均训练损失和准确率
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(self.train_loader, desc='Training') as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 梯度清零
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # 反向传播和优化
                loss.backward()
                self.optimizer.step()
                
                # 统计
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': running_loss / (pbar.n + 1),
                    'acc': 100. * correct / total
                })
        
        train_loss = running_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        
        return train_loss, train_acc
    
    def validate(self):
        """
        验证模型
        Returns:
            平均验证损失和准确率
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc='Validation') as pbar:
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # 前向传播
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    # 统计
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'loss': running_loss / (pbar.n + 1),
                        'acc': 100. * correct / total
                    })
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        
        return val_loss, val_acc
    
    def test(self):
        """
        测试模型
        Returns:
            测试准确率、混淆矩阵和分类报告
        """
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            with tqdm(self.test_loader, desc='Testing') as pbar:
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    # 前向传播
                    outputs = self.model(inputs)
                    _, predicted = outputs.max(1)
                    # 统计
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    # 保存预测和真实标签用于混淆矩阵
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        
        test_acc = 100. * correct / total
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        # 计算分类报告
        report = classification_report(
            all_labels, all_preds,
            target_names=['愤怒', '厌恶', '恐惧', '开心', '难过', '惊讶', '中性'],
            output_dict=True
        )
        return test_acc, cm, report
    
    def save_model(self, epoch, is_best=False):
        """
        保存模型
        Args:
            epoch: 当前epoch
            is_best: 是否为最佳模型
        """
        save_dir = self.config['paths']['checkpoint_dir']
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_score': self.best_score
        }
        
        # 保存当前模型
        checkpoint_path = os.path.join(save_dir, f'latest_resnet_epoch.pth')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Model saved at {checkpoint_path}")
        
        # 如果是最佳模型，保存为best_resnet_model.pth
        if is_best:
            best_model_path = os.path.join(save_dir, 'best_resnet_model.pth')
            torch.save(checkpoint, best_model_path)
            logger.info(f"Best model saved at {best_model_path}")
    
    def plot_training_history(self):
        plt.figure(figsize=(12, 5))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Training Accuracy')
        plt.plot(self.val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        # 保存图像
        plt.tight_layout()
        history_path = os.path.join(self.config['paths']['results_dir'], 'resnet_training_history.png')
        plt.savefig(history_path)
        logger.info(f"Training history saved at {history_path}")
        plt.close()
    
    def plot_confusion_matrix(self, cm):
        plt = setup_matplotlib(self.config)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['愤怒', '厌恶', '恐惧', '开心', '难过', '惊讶', '中性'],
            yticklabels=['愤怒', '厌恶', '恐惧', '开心', '难过', '惊讶', '中性']
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # 保存图像
        cm_path = os.path.join(self.config['paths']['results_dir'], 'resnet_confusion_matrix.png')
        plt.savefig(cm_path)
        logger.info(f"Confusion matrix saved at {cm_path}")
        plt.close()
    
    def save_classification_report(self, report):
        # 转换为DataFrame
        df = pd.DataFrame(report).transpose()
        # 保存到CSV
        report_path = os.path.join(self.config['paths']['results_dir'], 'resnet_classification_report.csv')
        df.to_csv(report_path)
        logger.info(f"Classification report saved at {report_path}")
    
    def train(self):
        """
        训练主函数
        """
        logger.info("Starting training...")
        # 预热学习率
        if self.config['train'].get('warmup_epochs', 0) > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.config['train']['warmup_epochs']
            )
        
        for epoch in range(self.config['train']['epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config['train']['epochs']}")
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch()
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            # 验证
            val_loss, val_acc = self.validate()
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            # 检查是否为最佳模型
            is_best = val_acc > self.best_score
            if is_best:
                self.best_score = val_acc
                self.early_stopping_counter = 0  # 重置早停计数器
                logger.info(f"New best validation accuracy: {self.best_score:.2f}%")
            else:
                # 更新早停计数器
                if self.early_stopping_enabled:
                    self.early_stopping_counter += 1
                    logger.info(f"Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                    # 检查是否触发早停
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
            # 保存模型
            self.save_model(epoch, is_best)
            # 更新学习率
            if epoch < self.config['train'].get('warmup_epochs', 0):
                warmup_scheduler.step() 
            elif self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau): #  Plateau 调度器
                    self.scheduler.step(val_loss)
                else: # 其他调度器
                    self.scheduler.step()
        
        # 绘制训练历史
        self.plot_training_history()
        # 测试模型
        logger.info("Testing model...")
        test_acc, cm, report = self.test()
        logger.info(f"Test Accuracy: {test_acc:.2f}%")
        # 绘制混淆矩阵
        self.plot_confusion_matrix(cm)
        # 保存分类报告
        self.save_classification_report(report)
        
        logger.info("Training completed!")

def main():
    """
    主函数
    """
    # 加载配置文件
    config_path = os.path.join('src', 'configs', 'config.yaml')
    config = load_config(config_path)

    # 更新配置以适应FER-2013数据集
    config['data']['dataset_name'] = 'FER-2013'
    config['data']['root_dir'] = 'data/FER_2013'  # 使用实际的数据集路径
    config['data']['image_size'] = 48  # 数据集实际图像尺寸为48x48
    config['data']['batch_size'] = 32
    config['data']['num_workers'] = 8
    config['model']['name'] = 'ResNet18'
    config['model']['num_classes'] = 7  # FER-2013有7个情绪类别
    config['model']['depth'] = '34'  # 使用ResNet-18，或者ResNet-34
    config['model']['zero_init_residual'] = True # 零初始化残差块的最后一个BN层
    
    # 增加权重衰减以减少过拟合
    config['train']['weight_decay'] = 0.01  # 从默认的较小值增加到0.01
    config['train']['optimizer'] = 'AdamW'  # 确保使用AdamW优化器
    config['train']['early_stopping_patience'] = 15  # 启用早停机制
    config['model']['dropout_rate'] = 0.5  # 设置Dropout率
    
    # 创建训练器
    trainer = ResNetTrainer(config)
    
    # 开始训练
    trainer.train()

if __name__ == '__main__':
    main()