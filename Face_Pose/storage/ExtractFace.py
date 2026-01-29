# 首先设置环境变量来控制ONNX Runtime和其他库的日志
import os
os.environ['ORT_LOG_LEVEL'] = 'ERROR'  # ONNX Runtime日志级别: ERROR
os.environ['ONNX_RUNTIME_LOG_LEVEL'] = 'ERROR'  # 另一个ONNX Runtime日志环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 如果使用TensorFlow相关库

# 导入其他库
import cv2
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import datetime
import io
from contextlib import redirect_stdout, redirect_stderr

# 导入insightface（放在日志配置之后）
from insightface.app import FaceAnalysis

# 导入面部情绪识别器
from Face_Pose.FaceEmotionRecognizer import FaceEmotionRecognizer

# 配置Python logging模块
logging.basicConfig(level=logging.ERROR)

# 控制特定模块的日志级别
logging.getLogger('insightface').setLevel(logging.ERROR)
logging.getLogger('onnxruntime').setLevel(logging.ERROR)
logging.getLogger('cv2').setLevel(logging.ERROR)
logging.getLogger('numpy').setLevel(logging.ERROR)


# 设置日志
log_dir = '/home/zdx/python_daima/MVim/MVim/logs/extractface/'
os.makedirs(log_dir, exist_ok=True)

# 生成日志文件名，包含日期
date_str = datetime.datetime.now().strftime('%Y%m%d')
log_file = os.path.join(log_dir, f'extractface_{date_str}.log')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('ExtractFace')

"""
提取照片中主驾的面部区域.基于RetinaFace模型.
ReinaFace人脸检测+5点关键点检测>>>bbox剪裁>>>输出face patch
无需训练,直接使用预训练模型.API 调用InsightFace.
"""
# ----------------------------------------
# 初始化人脸检测器
# ----------------------------------------
# 设置要使用的模型模块，排除性别和年龄识别模型
allowed_modules = ['detection', 'recognition', 'landmark_2d_106', 'landmark_3d_68'] # 检测、识别、2D关键点、3D关键点

# 使用上下文管理器捕获InsightFace初始化时的输出，减少终端噪音
with io.StringIO() as f, redirect_stdout(f), redirect_stderr(f):
    app = FaceAnalysis(
      name='buffalo_l',             # RetinaFace模型+landmark
      allowed_modules=allowed_modules,  # 排除性别和年龄识别模型
      providers=['CUDAExecutionProvider', 'CPUExecutionProvider']  # 优先使用GPU
      )
    app.prepare(ctx_id=0, det_size=(480, 640)) # ctx_id=0 表示使用GPU.ctx_id=-1 表示使用CPU.det_size=(640, 640) 表示输入图像的大小.

logger.info(f"人脸检测器初始化完成,使用GPU运行,使用模型:{allowed_modules}")

# ----------------------------------------
# 初始化面部情绪识别器
# ----------------------------------------
# ResEmoteNetV2模型训练权重路径
emotion_model_weight = '/home/zdx/python_daima/MVim/ResEmoteNetV2/outputs/checkpoints/best_model.pth'

# 初始化情绪识别器
with io.StringIO() as f, redirect_stdout(f), redirect_stderr(f):
    emotion_recognizer = FaceEmotionRecognizer(emotion_model_weight)

logger.info("面部情绪识别器初始化完成")

# ----------------------------------------
# 主驾人脸选择函数
# ----------------------------------------
def face_score(face):
  x1, y1, x2, y2 = face.bbox.astype(int)
  area = (x2 - x1) * (y2 - y1)
  center_x = (x1 + x2) / 2
  return area-0.5 * center_x

# ----------------------------------------
# 处理单张图像
# ----------------------------------------
def process_single_image(image_path, output_path='out_driver_face.jpg', return_emotion=True):
    """
    处理单张图像，提取主驾人脸并保存，可选返回情绪预测结果
    Args:
        image_path: 输入图像路径
        output_path: 输出人脸图像路径
        return_emotion: 是否返回情绪预测结果
    Returns:
        dict or None: 如果return_emotion为True，返回包含情绪预测结果的字典，否则返回None
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 人脸检测
    faces = app.get(image)
    if len(faces) == 0:
        raise ValueError(f"未检测到人脸: {image_path}")

    # 选择主驾人脸
    main_face = sorted(faces, key=face_score, reverse=True)[0]

    # 剪裁面部区域
    x1, y1, x2, y2 = main_face.bbox.astype(int)
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    face_crop = image[y1:y2, x1:x2]
    cv2.imwrite(output_path, face_crop)
    logger.info(f"主驾面部区域已保存为 {output_path}")
    
    # 情绪预测
    if return_emotion:
        emotion_result = emotion_recognizer.predict_emotion(face_crop)
        logger.info(f"主驾面部情绪预测结果: {emotion_result['emotion']} (置信度: {emotion_result['confidence']:.2f})")
        return emotion_result
    
    return None

# ----------------------------------------
# 批量处理图像
"""
from ExtractFace import batch_process_images
batch_process_images(batch_tensor, output_dir='test_batch_output/', is_tensor_batch=True)
"""
# ----------------------------------------
def batch_process_images(input_data, output_dir='MVim/Face/results/emotion/', is_tensor_batch=False, return_emotions=True):
    """
    批量处理图像，提取主驾人脸并保存，可选返回情绪预测结果
    Args:
        input_data: 输入数据,可以是图像目录路径或PyTorch张量批次
        output_dir: 输出人脸图像目录
        is_tensor_batch: 是否为PyTorch张量批次
        return_emotions: 是否返回情绪预测结果
    Returns:
        list: 如果return_emotions为True，返回包含每个图像情绪预测结果的字典列表，否则返回空列表
    """
    import torch
    import numpy as np
    
    # 存储所有情绪预测结果
    emotion_results = []

    if is_tensor_batch:
        # 处理PyTorch张量批次
        if not isinstance(input_data, torch.Tensor):
            raise ValueError("input_data 不是 PyTorch 张量")
        
        # 获取批次大小
        batch_size = input_data.shape[0]
        
        for i in range(batch_size):
            try:
                # 获取单张图像张量
                image_tensor = input_data[i]
                
                # 反归一化并转换为OpenCV格式
                # 假设张量已经过归一化: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image_tensor.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image_tensor.device)
                image_tensor = image_tensor * std + mean
                
                # 将张量转换为numpy数组
                image_np = image_tensor.cpu().numpy()
                
                # 转换通道顺序: (C, H, W) -> (H, W, C)
                image_np = np.transpose(image_np, (1, 2, 0))
                
                # 转换为8位无符号整数
                image_np = (image_np * 255).astype(np.uint8)
                
                # 转换色彩空间: RGB -> BGR
                image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                # 人脸检测
                faces = app.get(image)
                if len(faces) == 0:
                    logger.warning(f"批次 [{i}] 未检测到人脸")
                    continue

                # 选择主驾人脸
                main_face = sorted(faces, key=face_score, reverse=True)[0]
                
                # 剪裁面部区域
                x1, y1, x2, y2 = main_face.bbox.astype(int)
                h, w = image.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                face_crop = image[y1:y2, x1:x2]
                output_path = os.path.join(output_dir, f"batch_face_{i}.jpg")
                cv2.imwrite(output_path, face_crop)
                logger.info(f"批次 [{i}] 主驾面部区域已保存为 {output_path}")
                
                # 情绪预测
                if return_emotions:
                    try:
                        # 情绪预测
                        emotion_result = emotion_recognizer.predict_emotion(face_crop)
                        emotion_result['image_index'] = i
                        emotion_result['output_path'] = output_path
                        emotion_results.append(emotion_result)
                        logger.info(f"批次 [{i}] 情绪预测结果: {emotion_result['emotion']} (置信度: {emotion_result['confidence']:.2f})")
                    except Exception as e:
                        logger.error(f"批次 [{i}] 情绪预测时出错: {e}")
                
            except Exception as e:
                logger.error(f"处理批次 [{i}] 时出错: {e}")
                continue
    else:
        # 处理目录中的图像（保持原有功能）
        for frname in os.listdir(input_data):
            if not frname.lower().endswith((".jpg", ".png")): # 只处理jpg和png图像
                continue 
            
            image_path = os.path.join(input_data, frname)
            output_path = os.path.join(output_dir, frname)
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图像: {image_path}")
                continue

            # 人脸检测
            faces = app.get(image)
            if len(faces) == 0:
                logger.warning(f"未检测到人脸: {image_path}")
                continue

            # 选择主驾人脸
            main_face = sorted(faces, key=face_score, reverse=True)[0]
            
            # 剪裁面部区域
            x1, y1, x2, y2 = main_face.bbox.astype(int)
            h, w = image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            face_crop = image[y1:y2, x1:x2]
            cv2.imwrite(output_path, face_crop)
            logger.info(f"主驾面部区域已保存为 {output_path}")
            
            # 情绪预测
            if return_emotions:
                try:
                    emotion_result = emotion_recognizer.predict_emotion(face_crop)
                    emotion_result['image_path'] = image_path
                    emotion_result['output_path'] = output_path
                    emotion_results.append(emotion_result)
                    logger.info(f"图像 {frname} 情绪预测结果: {emotion_result['emotion']} (置信度: {emotion_result['confidence']:.2f})")
                except Exception as e:
                    logger.error(f"图像 {frname} 情绪预测时出错: {e}")
    
    # 返回情绪预测结果
    return emotion_results

# ----------------------------------------
# 示例用法
# ----------------------------------------
if __name__ == "__main__":
    # 处理单张图像示例
    single_image_path = '/home/zdx/python_daima/MVim/MVim/data/SFDDD/images/test/img_1.jpg' # 测试图像路径
    output_path = 'driver_face.jpg' # 输出图像路径
    process_single_image(single_image_path, output_path)
    
    # 批量处理示例
    # input_dir = "images/"
    # output_dir = "driver_faces/"
    # batch_process_images(input_dir, output_dir)
