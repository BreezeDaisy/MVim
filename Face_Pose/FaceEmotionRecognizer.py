import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import logging
from pathlib import Path

# 导入ResEmoteNetV2模型结构
from Face.resemotenet import ResEmoteNetV2

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceEmotionRecognizer:
    def __init__(self, model_weight_path, device=None):
        """
        初始化面部情绪识别器
        Args:
            model_weight_path (str): 模型权重文件路径
            device (str, optional): 运行设备 (cuda/cpu). 默认自动检测
        """
        # 检测设备
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 加载模型
        self.model = ResEmoteNetV2(num_classes=7).to(self.device)
        
        # 加载checkpoint
        checkpoint = torch.load(model_weight_path, map_location=self.device)
        
        # 检查是否是完整的checkpoint（包含model_state_dict等）
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 直接加载state_dict
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        logger.info(f"face_emotion_recognizer模型加载完成: {model_weight_path}")
        
        # 定义情绪类别
        self.emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
    def predict_emotion(self, face_image):
        """
        预测单张面部图像的情绪
        Args:
            face_image (numpy.ndarray): 输入面部图像 (BGR格式)
        Returns:
            dict: 包含情绪预测结果的字典
        """
        # 预测
        with torch.no_grad():
            outputs = self.model(face_image)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # 返回结果
        result = {
            'emotion': self.emotion_classes[predicted_class],
            'emotion_idx': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy().tolist(),
            'classes': self.emotion_classes
        }
        
        return result
    
    def batch_predict_emotions(self, face_images):   
        # 预测
        with torch.no_grad():
            outputs = self.model(face_images)
            probabilities = F.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()
            confidences = probabilities[torch.arange(probabilities.size(0)), predicted_classes].cpu().numpy()
        
        # 返回结果
        results = []
        for i in range(len(face_images)):
            result = {
                'emotion': self.emotion_classes[predicted_classes[i]],
                'emotion_idx': predicted_classes[i],
                'confidence': confidences[i],
                'probabilities': probabilities[i].cpu().numpy().tolist(),
                'classes': self.emotion_classes
            }
            results.append(result)
        
        return results

# 测试代码
if __name__ == '__main__':
    # 模型权重路径
    weight_path = '/home/zdx/python_daima/MVim/ResEmoteNetV2/outputs/checkpoints/best_model.pth'
    
    # 创建识别器实例
    recognizer = FaceEmotionRecognizer(weight_path)
    
    # 测试单张图像（需要替换为实际图像路径）
    test_image_path = '/home/zdx/python_daima/MVim/MVim/driver_face.jpg'
    if Path(test_image_path).exists():
        face_image = cv2.imread(test_image_path)
        result = recognizer.predict_emotion(face_image)
        logger.info(f"单张图像情绪预测结果: {result}")
    else:
        logger.warning(f"测试图像不存在: {test_image_path}")
    
    logger.info("FaceEmotionRecognizer初始化完成")