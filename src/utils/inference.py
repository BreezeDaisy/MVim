import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# 导入自定义模块
from src.models.mamba_model import create_mamba_model
from src.utils.utils import load_config, preprocess_image, postprocess_prediction

class InferenceEngine:
    """
    推理引擎类，用于加载模型并进行预测
    """
    def __init__(self, config_path=None, checkpoint_path=None):
        # 加载配置
        if config_path is None:
            config_path = 'src/configs/config.yaml'
        self.config = load_config(config_path)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = create_mamba_model(self.config)
        self.model.to(self.device)
        self.model.eval()
        
        # 加载权重
        if checkpoint_path is None:
            checkpoint_path = self.config['inference']['checkpoint_path']
        self.load_checkpoint(checkpoint_path)
        
        # 类别名称映射
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
        
        # 定义预处理变换
        self.transform = transforms.Compose([
            transforms.Resize((self.config['data']['image_size'], self.config['data']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_checkpoint(self, checkpoint_path):
        """加载模型权重"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"成功加载模型权重: {checkpoint_path}")
        except Exception as e:
            print(f"加载模型权重失败: {e}")
            print("将使用随机初始化的模型")
    
    def preprocess(self, image):
        """预处理输入图像"""
        # 如果是图像路径，先加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # 如果是PIL图像，应用变换
        if isinstance(image, Image.Image):
            image_tensor = self.transform(image)
        # 如果是numpy数组，转换为PIL图像再应用变换
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            image_tensor = self.transform(image)
        # 如果已经是张量，检查维度
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image_tensor = image
            else:
                raise ValueError(f"输入张量维度错误，应为[3, H, W]，实际为{image.shape}")
        else:
            raise TypeError(f"不支持的输入类型: {type(image)}")
        
        # 添加批次维度
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def predict(self, image):
        """进行预测"""
        # 预处理
        input_tensor = self.preprocess(image)
        
        # 前向传播
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # 后处理
        results = self.postprocess(output)
        
        return results
    
    def postprocess(self, output):
        """后处理模型输出"""
        # 应用softmax获取概率
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # 获取置信度和预测类别
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        # 构建结果字典
        results = {
            'predicted_class_index': predicted_idx.item(),
            'predicted_class': f'c{predicted_idx.item()}',
            'predicted_class_name': self.class_names.get(f'c{predicted_idx.item()}', 'Unknown'),
            'confidence': confidence.item(),
            'probabilities': {}
        }
        
        # 添加所有类别的概率
        for i in range(probabilities.shape[1]):
            class_key = f'c{i}'
            class_name = self.class_names.get(class_key, 'Unknown')
            results['probabilities'][class_key] = {
                'class_name': class_name,
                'probability': probabilities[0, i].item()
            }
        
        return results
    
    def batch_predict(self, images, batch_size=16):
        """批量预测"""
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            # 预处理批量图像
            batch_tensors = []
            for img in batch:
                # 去掉批次维度
                tensor = self.preprocess(img).squeeze(0)
                batch_tensors.append(tensor)
            
            # 堆叠成批次
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # 前向传播
            with torch.no_grad():
                outputs = self.model(batch_tensor)
            
            # 后处理每个样本
            for j in range(outputs.shape[0]):
                result = self.postprocess(outputs[j:j+1])
                results.append(result)
        
        return results
    
    def predict_with_visualization(self, image, visualize=True):
        """进行预测并可视化结果"""
        # 进行预测
        results = self.predict(image)
        
        if visualize:
            import matplotlib.pyplot as plt
            
            # 加载原始图像
            if isinstance(image, str):
                original_image = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                original_image = image
            elif isinstance(image, np.ndarray):
                original_image = Image.fromarray(image)
            else:
                # 如果是张量，尝试转换回图像
                if isinstance(image, torch.Tensor):
                    if len(image.shape) == 4:
                        image = image.squeeze(0)
                    original_image = transforms.ToPILImage()(image)
                else:
                    raise TypeError("无法可视化的输入类型")
            
            # 创建可视化
            plt.figure(figsize=(10, 8))
            
            # 绘制原始图像
            plt.subplot(2, 1, 1)
            plt.imshow(original_image)
            plt.title(f"预测结果: {results['predicted_class_name']} ({results['confidence']:.2%})")
            plt.axis('off')
            
            # 绘制概率分布
            plt.subplot(2, 1, 2)
            class_indices = list(range(len(self.class_names)))
            class_labels = [self.class_names[f'c{i}'] for i in class_indices]
            probabilities = [results['probabilities'][f'c{i}']['probability'] for i in class_indices]
            
            bars = plt.barh(class_labels, probabilities)
            # 为预测的类别添加不同的颜色
            for i, bar in enumerate(bars):
                if i == results['predicted_class_index']:
                    bar.set_color('red')
                else:
                    bar.set_color('blue')
            
            plt.xlabel('概率')
            plt.title('各类别概率分布')
            plt.xlim(0, 1)
            
            # 添加概率值标签
            for i, prob in enumerate(probabilities):
                plt.text(prob + 0.01, i, f'{prob:.2%}', va='center')
            
            plt.tight_layout()
            plt.show()
        
        return results
    
    def get_top_k_predictions(self, image, k=3):
        """获取前k个预测结果"""
        results = self.predict(image)
        
        # 获取所有类别的概率
        all_probs = []
        for class_key, info in results['probabilities'].items():
            all_probs.append((class_key, info['class_name'], info['probability']))
        
        # 按概率排序
        all_probs.sort(key=lambda x: x[2], reverse=True)
        
        # 返回前k个
        top_k = all_probs[:k]
        
        return [{
            'class': item[0],
            'class_name': item[1],
            'probability': item[2]
        } for item in top_k]

def demo_inference():
    """演示推理功能"""
    # 创建推理引擎
    engine = InferenceEngine()
    
    print("推理引擎初始化完成！")
    print("可用的功能:")
    print("1. predict(image): 预测单个图像")
    print("2. batch_predict(images): 批量预测")
    print("3. predict_with_visualization(image): 预测并可视化")
    print("4. get_top_k_predictions(image, k=3): 获取前k个预测结果")
    
    # 如果有示例图像，可以进行演示
    # 这里使用随机生成的图像进行演示
    import numpy as np
    from PIL import Image
    
    # 生成随机图像
    random_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    random_image = Image.fromarray(random_image)
    
    print("\n使用随机生成的图像进行演示:")
    results = engine.predict(random_image)
    print(f"预测类别: {results['predicted_class_name']}")
    print(f"置信度: {results['confidence']:.2%}")
    
    print("\n前3个预测结果:")
    top_k = engine.get_top_k_predictions(random_image, k=3)
    for i, pred in enumerate(top_k):
        print(f"{i+1}. {pred['class_name']}: {pred['probability']:.2%}")

if __name__ == '__main__':
    demo_inference()