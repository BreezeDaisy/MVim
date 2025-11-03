import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

class SFDataset(Dataset):
    """
    SFDDD数据集加载器
    SFDDD (State Farm Distracted Driver Detection) 数据集包含10个类别
    类别标签: c0 - 安全驾驶, c1 - 右手打字, c2 - 右手打电话, c3 - 左手打字, 
             c4 - 左手打电话, c5 - 调收音机, c6 - 喝水, c7 - 拿后面的东西, 
             c8 - 整理头发/化妆, c9 - 和乘客说话
    """
    def __init__(self, root_dir, split='train', transform=None, image_size=224):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # 类别映射
        self.class_names = {
            'c0': '安全驾驶',
            'c1': '右手打字',
            'c2': '右手打电话',
            'c3': '左手打字',
            'c4': '左手打电话',
            'c5': '调收音机',
            'c6': '喝水',
            'c7': '拿后面的东西',
            'c8': '整理头发/化妆',
            'c9': '和乘客说话'
        }
        
        # 类别到索引的映射
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names.keys())}
        
        # 获取图像路径和标签
        self.image_paths = []
        self.labels = []
        self._load_data()
        
        # 默认数据增强和预处理
        if self.transform is None:
            if self.split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    
    def _load_data(self):
        """加载数据集"""
        if self.split in ['train', 'val', 'test']:
            # 训练集包含在images/train目录下的类别文件夹
            train_dir = os.path.join(self.root_dir, 'images', 'train')
            
            # 尝试使用driver_imgs_list.csv获取标签信息
            csv_path = os.path.join(self.root_dir, 'driver_imgs_list.csv')
            use_csv = os.path.exists(csv_path)
            
            if use_csv:
                import pandas as pd
                df = pd.read_csv(csv_path)
                
                for _, row in df.iterrows():
                    class_name = row['classname']
                    img_name = row['img']
                    img_path = os.path.join(train_dir, class_name, img_name)
                    
                    if os.path.exists(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])
            else:
                # 如果没有csv文件，直接从文件夹结构加载
                for class_name in self.class_names.keys():
                    class_dir = os.path.join(train_dir, class_name)
                    if os.path.exists(class_dir):
                        for img_name in os.listdir(class_dir):
                            if img_name.endswith(('.jpg', '.png')):
                                img_path = os.path.join(class_dir, img_name)
                                self.image_paths.append(img_path)
                                self.labels.append(self.class_to_idx[class_name])
            
            # 划分训练集、验证集和测试集
            if self.split in ['train', 'val', 'test']:
                from sklearn.model_selection import train_test_split
                X_train_val, X_test, y_train_val, y_test = train_test_split(
                    self.image_paths, self.labels, test_size=0.15, stratify=self.labels, random_state=42
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42
                )
                
                if self.split == 'train':
                    self.image_paths, self.labels = X_train, y_train
                elif self.split == 'val':
                    self.image_paths, self.labels = X_val, y_val
                else:  # test
                    self.image_paths, self.labels = X_test, y_test
        else:
            # 测试集位于images/test目录
            test_dir = os.path.join(self.root_dir, 'images', 'test')
            if os.path.exists(test_dir):
                for img_name in os.listdir(test_dir):
                    if img_name.endswith(('.jpg', '.png')):
                        img_path = os.path.join(test_dir, img_name)
                        self.image_paths.append(img_path)
                        # 测试集可能没有标签，设置为-1
                        self.labels.append(-1)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个空白图像和标签
            image = Image.new('RGB', (self.image_size, self.image_size))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_dataloaders(config):
    """
    获取数据加载器
    """
    # 创建数据集
    train_dataset = SFDataset(
        root_dir=config['data']['root_dir'],
        split='train',
        image_size=config['data']['image_size']
    )
    
    val_dataset = SFDataset(
        root_dir=config['data']['root_dir'],
        split='val',
        image_size=config['data']['image_size']
    )
    
    test_dataset = SFDataset(
        root_dir=config['data']['root_dir'],
        split='test',
        image_size=config['data']['image_size']
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
    
    return train_loader, val_loader, test_loader, train_dataset.class_names

# 如果没有实际数据，创建一个虚拟数据集用于演示
class DummyDriverDataset(Dataset):
    """虚拟数据集，用于无实际数据时的代码测试"""
    def __init__(self, num_samples=100, image_size=224):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = 10
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.class_names = {
            'c0': '安全驾驶',
            'c1': '右手打字',
            'c2': '右手打电话',
            'c3': '左手打字',
            'c4': '左手打电话',
            'c5': '调收音机',
            'c6': '喝水',
            'c7': '拿后面的东西',
            'c8': '整理头发/化妆',
            'c9': '和乘客说话'
        }
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机图像
        image = torch.rand(3, self.image_size, self.image_size)
        # 生成随机标签
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label

def get_dummy_dataloaders(config):
    """获取虚拟数据加载器"""
    train_dataset = DummyDriverDataset(num_samples=100, image_size=config['data']['image_size'])
    val_dataset = DummyDriverDataset(num_samples=30, image_size=config['data']['image_size'])
    test_dataset = DummyDriverDataset(num_samples=20, image_size=config['data']['image_size'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    return train_loader, val_loader, test_loader, train_dataset.class_names