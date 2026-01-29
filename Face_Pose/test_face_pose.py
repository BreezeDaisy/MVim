# 首先配置日志，确保在所有导入和使用之前初始化
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', # 包含时间、日志级别和消息
    handlers=[
        logging.FileHandler("/home/zdx/python_daima/MVim/MVim/logs/test_face_pose.log"), # 日志文件路径
        logging.StreamHandler() # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__) # 获取当前模块的日志记录器

# 然后导入其他模块
import torch
import torch.nn as nn
from ultralytics import YOLO
import torch.nn.functional as F
# from deepface import DeepFace
import yaml
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from src.data.dataset import get_dataloaders # 数据集划分
from Face_Pose.FaceEmotionRecognizer import FaceEmotionRecognizer # 情绪识别模型
from src.models.mamba_model import create_mamba_model # 行为识别模型

"""
Dataloader -> images [B,3,H,W] |---> Face Branch: face detect -> face crop -> face net -> emotion recognizer : face latent
                               L-- > Pose Branch: pose detect -> pose regressor : pose latent
"""

"""
Face detector and batch crop based on YOLOv8-Face
"""
def load_face_detector(weight_path, device='cuda'):
  model = YOLO(weight_path)
  logger.info(f"人脸检测模型加载完成: {weight_path}")
  model = model.to(device).eval()
  return model

@torch.no_grad()
def batch_face_crop(images,weight_path,size=224,conf=0.5): # conf表示置信度阈值，size表示目标人脸尺寸
  """
  对一批图像进行人脸检测和裁剪.
  images: 输入图像 batch, 形状为 [B, 3, H, W]
  conf: 置信度阈值, 默认值为 0.5
  size: 目标人脸尺寸, 默认值为 224
  返回: 剪裁后的face框, 形状为[B, 3, size, size]
  """
  face_detector = load_face_detector(weight_path) 
  
  # 对标准化后的图像进行反标准化处理
  mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
  images_denorm = images * std + mean # 反标准化至[0,1]
  
  # 确保图像在[0, 1]范围内
  images_denorm = torch.clamp(images_denorm, 0, 1)

  # 准备数据加载器
  results = face_detector(images_denorm, conf=conf, verbose=False)
  
  # 统计检测结果
  total_faces_detected = sum(1 for r in results if r.boxes is not None and len(r.boxes) > 0)
  logger.info(f"人脸检测结果统计 - 总图像数: {len(images)}, 检测到人脸的图像数: {total_faces_detected}")

  faces = []
  for i, r in enumerate(results):
    if r.boxes is None or len(r.boxes) == 0:
      # 如果没有检测到人脸, 则返回1通道的全零张量。处理方式暂定，为了保证批次数量的稳定
      logger.warning(f"未检测到人脸，样本索引: {i}")
      faces.append(torch.zeros((1, size, size), dtype=torch.float32))
      continue
    boxes = r.boxes.xyxy
    scores = r.boxes.conf
    # logger.info(f"检测到人脸数量: {len(boxes)}, 最高置信度: {scores.max():.4f}")
    idx = scores.argmax() # 取置信度最高的框。模型检测的不一定是单人脸，但是输入只有一个人，所以取置信度最高的框即可
    x1, y1, x2, y2 = boxes[idx].long() # 转换为整数坐标
    
    _, _, H, W = images.shape
    
    # 方法：以检测框中心为基准，扩展到固定尺寸
    # 计算检测框的中心坐标
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    # 计算固定尺寸框的半边长
    half_size = size // 2
    # 计算新的边界坐标
    new_x1 = cx - half_size
    new_y1 = cy - half_size
    new_x2 = new_x1 + size
    new_y2 = new_y1 + size
    # 处理边界溢出情况
    # 创建padding参数
    pad_left = max(0, -new_x1)
    pad_top = max(0, -new_y1)
    pad_right = max(0, new_x2 - W)
    pad_bottom = max(0, new_y2 - H)
    # 调整坐标到图像边界内
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(W, new_x2)
    new_y2 = min(H, new_y2)
    # 从原始图像中裁剪
    face = images[i, :, new_y1:new_y2, new_x1:new_x2]
    # 如果有边界溢出，使用0进行padding
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
      face = F.pad(face, (pad_left, pad_right, pad_top, pad_bottom))

    # 将3通道RGB图像转换为1通道灰度图像。情绪识别模型的训练时依据灰度图像，这里暂时将输入转为灰度图像，未来再对其进行调整。
    # 使用RGB到灰度的转换公式: Y = 0.299R + 0.587G + 0.114B
    gray_face = 0.299 * face[0, :, :] + 0.587 * face[1, :, :] + 0.114 * face[2, :, :]
    # 添加通道维度，从[H, W]变为[1, H, W]
    gray_face = gray_face.unsqueeze(0)

    faces.append(gray_face) 
    # logger.info(f"人脸图像的灰度图像形状: {gray_face.shape}")

  # 将列表转换为PyTorch张量，形状为[B, 1, size, size]
  return torch.stack(faces)

"""
Face分支: 人脸检测+情绪识别
"""
class FaceBranch(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.device = config['inference']['device']
    self.face_detector_weights = config['inference']['face_detector_weights']
    self.emotion_model_weights = config['inference']['emotion_recognizer_weights']
    
    # 初始化情绪识别模型
    self.emotion_recognizer = FaceEmotionRecognizer(self.emotion_model_weights, device=self.device)
  
  def forward(self, images):
    """
    前向传播：人脸检测 -> 情绪识别
    Args:
        images: 输入图像批次，形状为[B, 3, H, W]
    Returns:
        tuple:
            logits: 情绪预测的logits,形状为[B, 7]
            latent: 情绪特征，形状为[B, 128]
    """
    # 1. 批量检测人脸
    size = self.config['data']['face_size']
    faces = batch_face_crop(images, self.face_detector_weights,size=size)
    faces = faces.to(self.device)
    
    # 2. batch_face_crop返回的faces已经是[B, 1, size, size]的PyTorch张量，符合情绪识别模型输入要求
    emotion_results = self.emotion_recognizer.batch_predict_emotions(faces)
    
    # 4. 准备输出格式（保持与原始接口兼容）
    batch_size = images.shape[0]
    
    # 初始化logits和latent
    logits = torch.zeros(batch_size, 7, device=self.device)
    # latent = torch.zeros(batch_size, 128, device=self.device)
    
    # 从情绪识别结果中提取信息
    for i, result in enumerate(emotion_results):
        if i < batch_size:
            # 将概率转换为logits
            probs = torch.tensor(result['probabilities'], device=self.device)
            logits[i] = probs.log()  # logits = log(probs)
            
            # 使用概率作为latent特征的一部分
            # 设计从模型中获取latent feature的方法
          
    return logits, faces

class PoseBranch(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.config = config
    self.device = config['inference']['device']
    self.model = None
    self.pose_model_weights = config['inference']['checkpoint_path']
    self.load_model()
    
  def load_model(self):
    """加载训练好的模型权重"""
    self.model = create_mamba_model(self.config).to(self.device)
    # 加载checkpoint
    checkpoint = torch.load(self.pose_model_weights, map_location=self.device)
    # 检查是否是完整的checkpoint（包含model_state_dict等）
    if 'model_state_dict' in checkpoint:
        self.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 直接加载state_dict
        self.model.load_state_dict(checkpoint)
        
    self.model.eval()
    logger.info(f"pose模型加载完成: {self.pose_model_weights}")
    return self.model
  
  def forward(self, images):
    """前向传播,接收images tensor作为输入"""
    if self.model is None:
      raise RuntimeError("模型未加载,请先调用load_model方法")
    
    with torch.no_grad(): # 禁用梯度计算，提高推理速度
      # 预处理：将图像移动到正确的设备
      processed_images = images.to(self.device)
      # 将处理后的图像输入模型
      logits = self.model(processed_images)

      # 使用概率作为latent特征的一部分
      # 设计从模型中获取latent feature的方法
            
    return logits

#===================================================#
# face intent 和 pose intent 合并，留作接口
#===================================================#
class IntentEncoder(nn.Module):
  def __init__(self,in_dim=256,intent_dim=64):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(in_dim,128),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(128,intent_dim)
    )

  def forward(self,face_intent,pose_intent):
    x = torch.cat([face_intent,pose_intent],dim=-1) # 在最后一个维度上拼接
    return self.net(x)

#===================================================#
# 驾驶员状态预测模型
#===================================================#
class DriverPerceptionModel(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.config = config
    self.face_branch = FaceBranch(config)
    self.pose_branch = PoseBranch(config)
    self.intent_encoder = IntentEncoder()

  def forward(self,images):
    # face_logits, face_intent = self.face_branch(images)
    # pose_logits, pose_intent = self.pose_branch(images)
    face_logits, faces = self.face_branch(images)
    pose_logits = self.pose_branch(images)
    # intent = self.intent_encoder(face_intent.detach(),pose_intent.detach())

    return {
      'face_logits': face_logits,
      'pose_logits': pose_logits,
      'faces': faces,
      # 'face_intent': face_intent,
      # 'pose_intent': pose_intent,
      # 'intent': intent  
    }

if __name__ == '__main__':
  
  # 创建results目录
  results_dir = '/home/zdx/python_daima/MVim/MVim/Face_Pose/results/260123'
  os.makedirs(results_dir, exist_ok=True)
  
  # 情绪类别和姿态类别
  emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
  pose_classes = ['safe driving', 'right hand on phone', 'right hand calling', 
                  'left hand on phone', 'left hand calling', 'adjusting radio', 
                  'drinking', 'reaching behind', 'doing hair/makeup', 'talking to passenger']
  
  with open('src/configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
  
  # 1. 加载数据
  train_loader, val_loader, test_loader, class_names = get_dataloaders(config)
  
  # 2. 初始化驾驶员感知模型
  model = DriverPerceptionModel(config)
  model.to(config['inference']['device'])
  
  # 3. 进行推理并可视化
  with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
      logger.info(f"输入图像批次形状: {images.shape}")
      
      # 执行推理
      results = model(images)
      
      # 打印结果形状
      logger.info(f"人脸logits形状: {results['face_logits'].shape}")
      logger.info(f"姿态logits形状: {results['pose_logits'].shape}")
      logger.info(f"提取的面部区域形状: {results['faces'].shape}")
      
      # 选择5张随机图像进行可视化
      visualize_count = 5
      num_images = len(images)
      if num_images > visualize_count:
          random_indices = np.random.choice(num_images, visualize_count, replace=False)
      else:
          random_indices = range(num_images)
      
      for i in random_indices:
        # 获取原始图像
        img = images[i].cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        
        # 反标准化图像以便可视化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = (img * 255).astype(np.uint8)  # 转换为[0, 255]范围的uint8
        
        # 获取预测结果
        face_logits = results['face_logits'][i].cpu().numpy()
        pose_logits = results['pose_logits'][i].cpu().numpy()
        
        # 计算概率
        face_probs = F.softmax(torch.from_numpy(face_logits), dim=0).numpy()
        pose_probs = F.softmax(torch.from_numpy(pose_logits), dim=0).numpy()
        
        # 获取预测的类别
        predicted_face = np.argmax(face_probs)
        predicted_pose = np.argmax(pose_probs)
        
        # 真实标签
        true_label = labels[i].item()
        
        # 获取人脸提取区域
        face_img = results['faces'][i]  # 形状为 [1, H, W]
        face_img = face_img.squeeze(0).cpu().numpy()  # 转换为 [H, W]
        
        # 对人脸图像进行归一化处理，确保在[0, 1]范围内显示
        if face_img.max() > face_img.min():
            # 归一化到[0, 1]范围
            face_img_normalized = (face_img - face_img.min()) / (face_img.max() - face_img.min())
        else:
            # 如果所有值都相同（包括全零），保持不变
            face_img_normalized = face_img.copy()
        
        # 创建可视化图像（修改为三列布局：原图+人脸 | 情绪识别 | 行为识别）
        plt.figure(figsize=(15, 8))
        
        # 列1：原图和人脸（按真实比例显示）
        # 创建子图网格，增加原图与人脸的高度比例
        grid = plt.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[4, 1])
        
        # 显示原图
        ax1 = plt.subplot(grid[0, 0])
        ax1.imshow(img)
        ax1.set_title(f"Sample {i+1}")
        ax1.axis('off') # 关闭坐标轴
        # ax1.set_aspect('equal')  # 保持正方形比例
        
        # 显示人脸提取区域
        ax2 = plt.subplot(grid[1, 0])
        ax2.imshow(face_img_normalized, cmap='gray', extent=[0, 12, 0, 12])
        ax2.set_title("Extracted Face")
        ax2.axis('off') # 关闭坐标轴
        # 设置人脸区域的aspect ratio为原始比例
        # ax2.set_aspect('equal')
        # 设置x和y轴的范围，确保人脸按真实大小显示
        # ax2.set_xlim(0, 24)
        # ax2.set_ylim(0, 24)
        
        # 列2：情绪识别结果
        ax3 = plt.subplot(grid[:, 1])
        ax3.barh(emotion_classes, face_probs)
        ax3.set_title(f"Emotion Prediction\n{emotion_classes[predicted_face]} ({face_probs[predicted_face]:.2f})")
        ax3.set_xlabel('Probability')
        ax3.set_xlim(0, 1)
        
        # 列3：姿态预测结果
        ax4 = plt.subplot(grid[:, 2])
        ax4.barh(pose_classes, pose_probs)
        ax4.set_title(f"Pose Prediction\n{pose_classes[predicted_pose]} ({pose_probs[predicted_pose]:.2f})\nTrue: {pose_classes[true_label]}")
        ax4.set_xlabel('Probability')
        ax4.set_xlim(0, 1)
        
        plt.tight_layout()
        
        # 保存可视化结果
        save_path = os.path.join(results_dir, f"result_{batch_idx}_{i}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"保存可视化结果: {save_path}")
      
      # 保存完整的推理结果
      batch_results = {
          'images': images.cpu().numpy(),
          'labels': labels.cpu().numpy(),
          'faces': results['faces'].cpu().numpy(),
          'face_logits': results['face_logits'].cpu().numpy(),
          'pose_logits': results['pose_logits'].cpu().numpy()
      }
      
      # 保存推理结果为npz文件
      npz_save_path = os.path.join(results_dir, f"batch_{batch_idx}_results.npz")
      np.savez_compressed(npz_save_path, **batch_results)
      logger.info(f"保存推理结果: {npz_save_path}")
      
      break  # 只处理一个批次
  
  # 5. 进行评估（可选）
  print("\n开始评估...")
  # 注意：这里可以添加完整的评估逻辑，如计算准确率、混淆矩阵等
  print("评估完成！")
  print(f"所有可视化结果已保存到: {results_dir}")
  logger.info(f"所有可视化结果已保存到: {results_dir}")

  

  

