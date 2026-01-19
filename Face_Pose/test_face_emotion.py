import os
import cv2
from ExtractFace import process_single_image, batch_process_images
import torch

# 测试单张图像的情绪识别
print("测试单张图像的情绪识别...")
single_image_path = '/home/zdx/python_daima/MVim/MVim/data/SFDDD/images/test/img_92052.jpg' # 测试图像路径
output_path = 'test_driver_face.jpg' # 输出图像路径

if os.path.exists(single_image_path):
    result = process_single_image(single_image_path, output_path)
    print(f"单张图像情绪识别结果: {result}")
else:
    print(f"测试图像不存在: {single_image_path}")

# # 测试批量张量的情绪识别
# print("\n测试批量张量的情绪识别...")
# # 生成虚拟张量批次 (batch_size=2, 3, 224, 224)
# batch_tensor = torch.rand(2, 3, 224, 224)
# output_dir = 'test_batch_output/'

# results = batch_process_images(batch_tensor, output_dir, is_tensor_batch=True)
# print(f"批量张量情绪识别结果: {results}")

# # 测试目录图像的情绪识别
# print("\n测试目录图像的情绪识别...")
# input_dir = '/home/zdx/python_daima/MVim/MVim/data/SFDDD/images/test/' # 测试图像目录
# output_dir = 'test_dir_output/'

# if os.path.exists(input_dir):
#     results = batch_process_images(input_dir, output_dir, is_tensor_batch=False)
#     print(f"目录图像情绪识别结果: {results}")
# else:
#     print(f"测试目录不存在: {input_dir}")

# print("\n所有测试完成！")
