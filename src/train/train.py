import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 导入自定义模块
from src.models.mamba_model import create_mamba_model
from src.data.dataset import get_dataloaders # get_dummy_dataloaders

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', # 包含时间、日志级别和消息
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Trainer:
    """
    训练器类,用于训练和评估Mamba模型
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = create_mamba_model(config).to(self.device)
        
        # 创建损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 创建学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 准备数据加载器
        self._prepare_data()
        
        # 初始化训练记录
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_score = 0.0  # 使用best_score替代best_val_acc以保持一致性
        
        # 创建所有必要的保存目录
        for dir_path in ['checkpoint_dir', 'logs_dir', 'results_dir']:
            if dir_path in config['paths']:
                os.makedirs(config['paths'][dir_path], exist_ok=True)

        os.makedirs(config['paths']['results_dir'], exist_ok=True)
    
    def _create_optimizer(self):
        """创建优化器"""
        if self.config['train']['optimizer'] == 'AdamW': # AdamW优化器
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['train']['learning_rate'],
                weight_decay=self.config['train']['weight_decay']
            )
        elif self.config['train']['optimizer'] == 'Adam': # Adam优化器 
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['train']['learning_rate'],
                weight_decay=self.config['train']['weight_decay'] 
            )
        else:
            optimizer = optim.SGD( # 随机梯度下降优化器
                self.model.parameters(),
                lr=self.config['train']['learning_rate'],
                momentum=0.9, # 动量因子
                weight_decay=self.config['train']['weight_decay']
            )
        return optimizer
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config['train']['scheduler'] == 'cosine': # 余弦退火学习率调度器
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['train']['epochs'],
                eta_min=0
            )
        elif self.config['train']['scheduler'] == 'step': # 步长学习率调度器
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config['train']['scheduler'] == 'reduce_on_plateau': # 基于验证损失的学习率衰减
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.1,
                patience=5
            )
        else:
            scheduler = None
        return scheduler
    
    def _prepare_data(self):
        """准备数据加载器"""
        try:
            # 尝试加载真实数据.这里将train目录下的22424张图片划分为15695(train)+3365(val)+3364(test)
            self.train_loader, self.val_loader, self.test_loader, self.class_names = get_dataloaders(self.config)
            logger.info(f"成功加载真实数据集，训练集大小: {len(self.train_loader.dataset)}, 验证集大小: {len(self.val_loader.dataset)}, 测试集大小: {len(self.test_loader.dataset)}")
        except Exception as e:
            logger.warning(f"加载真实数据失败: {e}，数据集不存在")
            # # 使用虚拟数据集
            # self.train_loader, self.val_loader, self.test_loader, self.class_names = get_dummy_dataloaders(self.config)
            # logger.info(f"已创建虚拟数据集用于演示，训练集大小: {len(self.train_loader.dataset)}, 验证集大小: {len(self.val_loader.dataset)}")
    
    def train_one_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # 梯度清零
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            self.optimizer.step()
            
            # 统计损失和准确率
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            progress_bar.set_description(f'Epoch [{epoch+1}/{self.config["train"]["epochs"]}]')
            progress_bar.set_postfix(loss=running_loss/total, acc=100.*correct/total)
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """验证模型性能"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validation'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0) # 统计总样本数
                correct += (predicted == labels).sum().item()
                
                # 保存所有标签和预测结果用于混淆矩阵
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        self.val_losses.append(epoch_loss)
        self.val_accs.append(epoch_acc)
        
        return epoch_loss, epoch_acc, all_labels, all_preds
    
    def test(self):
        """测试模型性能"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc='Testing'):
                # 跳过无标签的测试样本
                if labels[0] == -1:
                    continue
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # 获取概率分布
                probs = nn.functional.softmax(outputs, dim=1)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        if total == 0:
            logger.warning("测试集没有有效标签")
            return None
        
        test_loss = running_loss / total
        test_acc = correct / total
        
        # 生成混淆矩阵
        self._plot_confusion_matrix(all_labels, all_preds)
        
        # 生成分类报告
        self._generate_classification_report(all_labels, all_preds)
        
        logger.info(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        
        return test_loss, test_acc
    
    def _plot_confusion_matrix(self, true_labels, pred_labels):
        """生成混淆矩阵"""
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # 添加类别名称（使用原始类别标签）
        class_indices = list(range(len(self.class_names)))
        class_labels = [f'c{i}' for i in class_indices]  # 使用原始类别标签如c0, c1等
        plt.xticks(class_indices, class_labels, rotation=45, ha='right')
        plt.yticks(class_indices, class_labels, rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['paths']['results_dir'], 'confusion_matrix.png'))
        plt.close()
    
    def _generate_classification_report(self, true_labels, pred_labels):
        """生成分类报告"""
        report = classification_report(
            true_labels, 
            pred_labels,
            target_names=[f'c{i}' for i in range(len(self.class_names))],  # 使用原始类别标签
            output_dict=True
        )
        
        # 保存为CSV文件
        import pandas as pd
        df = pd.DataFrame(report).transpose()
        df.to_csv(os.path.join(self.config['paths']['results_dir'], 'classification_report.csv'))
    
    def save_model(self, epoch, is_best=False):
        """保存模型"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_acc': self.val_accs[-1],
            'config': self.config,
            'best_score': self.val_accs[-1] if is_best else (getattr(self, 'best_score', 0.0))
        }
        
        # 创建checkpoints目录
        checkpoint_dir = self.config['paths']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存最新模型
        latest_path = os.path.join(checkpoint_dir, 'latest_model.pth')
        torch.save(checkpoint, latest_path)
        logger.info(f"保存最新模型到: {latest_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型，验证准确率: {self.val_accs[-1]:.4f}，路径: {best_path}")
            # 更新最佳分数
            self.best_score = self.val_accs[-1]
    
    def load_model(self, checkpoint_path):
        """加载模型"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info(f"成功加载模型: {checkpoint_path}")
            # 返回加载信息，包含epoch和best_score
            return {
                'success': True,
                'epoch': checkpoint.get('epoch', 0),
                'best_score': checkpoint.get('best_score', checkpoint.get('val_acc', 0.0))
            }
        else:
            logger.info(f"模型文件不存在: {checkpoint_path}，重新训练模型")
            return {'success': False}
    
    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(12, 5))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot([acc * 100 for acc in self.train_accs], label='Train Acc')
        plt.plot([acc * 100 for acc in self.val_accs], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['paths']['results_dir'], 'training_history.png'))
        plt.close()
    
    def train(self):
        """训练主循环"""
        start_time = time.time()
        
        # 初始化最佳分数
        self.best_score = 0.0
        start_epoch = 0
        
        # 检查是否需要加载预训练模型
        checkpoint_path = self.config['model'].get('model_checkpoints', '')
        if checkpoint_path:
            load_info = self.load_model(checkpoint_path)
            if load_info and isinstance(load_info, dict) and load_info.get('success'):
                logger.info(f"从检查点恢复训练: {checkpoint_path}")
                # 恢复训练轮数和最佳分数
                start_epoch = load_info.get('epoch', 0)
                self.best_score = max(load_info.get('best_score', 0.0), self.best_score)
            else:
                logger.info("预训练模型加载失败，开始新的训练")
        else:
            logger.info("未指定预训练模型路径，开始新的训练")
        
        for epoch in range(start_epoch, self.config['train']['epochs']):
            # 训练一个epoch
            train_loss, train_acc = self.train_one_epoch(epoch)
            
            # 验证
            val_loss, val_acc, _, _ = self.validate()
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # 记录日志
            logger.info(f'Epoch [{epoch+1}/{self.config["train"]["epochs"]}], '
                        f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                        f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # 保存模型
            is_best = val_acc > self.best_score
            if is_best:
                self.best_score = val_acc
            self.save_model(epoch, is_best)
            
            # 绘制训练历史
            if (epoch + 1) % 5 == 0 or epoch == self.config['train']['epochs'] - 1:
                self.plot_training_history()
        
        end_time = time.time()
        logger.info(f"训练完成！总耗时: {(end_time - start_time) / 3600:.2f} 小时")
        logger.info(f"最佳验证准确率: {self.best_score:.4f}")
        
        # 最终测试
        logger.info("开始测试模型性能...")
        self.load_model(os.path.join(self.config['paths']['checkpoint_dir'], 'best_model.pth'))
        self.test()
        
        # 绘制最终训练历史
        self.plot_training_history()

def main():
    """主函数"""
    # 加载配置
    with open('src/configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建训练器并开始训练
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()