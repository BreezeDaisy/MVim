import numpy as np

# 加载NPZ文件
data = np.load('/home/zdx/python_daima/MVim/MVim/Face/results/batch_0_results.npz')

# 访问其中的数据
images = data['images']
labels = data['labels']
face_logits = data['face_logits']
pose_logits = data['pose_logits']

# 查看数据形状
print(f"图像形状: {images.shape}")
print(f"标签形状: {labels.shape}")
print(f"人脸logits形状: {face_logits.shape}")
print(f"姿态logits形状: {pose_logits.shape}")