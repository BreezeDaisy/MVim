import torch
import sys
# 添加项目路径
sys.path.append('/home/zdx/python_daima/MVim')
# 加载权重文件
checkpoint = torch.load('/home/zdx/python_daima/MVim/ResEmoteNetV2/outputs/checkpoints/best_model.pth', map_location='cpu')
# 检查是否有model_state_dict
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint
# 打印所有层名称
print('权重文件中的层名称：')
for key in state_dict.keys():
    print(f'  {key}')
print(f'\n总层数：{len(state_dict.keys())}')