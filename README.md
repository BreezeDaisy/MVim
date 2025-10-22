# MVim
基于mamba模型

# 驾驶员分心检测项目

## 项目介绍

本项目实现了基于三阶段Mamba架构的驾驶员分心检测系统。该系统能够输入驾驶员的姿势图像，输出对应的分心行为类别标签。项目使用SFDDD（State Farm Distracted Driver Detection）数据集进行训练和评估。

## 目录结构

```
├── src/
│   ├── configs/         # 配置文件
│   ├── data/            # 数据处理模块
│   ├── models/          # 模型定义
│   ├── train/           # 训练脚本
│   └── utils/           # 工具函数
├── data/                # 数据集目录
├── results/             # 结果保存目录
├── logs/                # 日志目录
├── checkpoints/         # 模型权重保存目录
├── requirements.txt     # 依赖列表
└── README.md            # 项目说明
```

## 环境要求

- Python 3.10.18
- PyTorch 2.9.0+cu126
- CUDA 12.6
- 

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据集准备

本项目使用SFDDD数据集。数据集包含10个类别，分别是：

- c0: 安全驾驶
- c1: 右手打字
- c2: 右手打电话
- c3: 左手打字
- c4: 左手打电话
- c5: 调收音机
- c6: 喝水
- c7: 拿后面的东西
- c8: 整理头发/化妆
- c9: 和乘客说话

### 数据格式

数据集应按照以下格式组织：

```
data/SFDDD/
├── c0/
│   ├── img_1.jpg
│   ├── img_2.jpg
│   └── ...
├── c1/
├── ...
└── c9/
```

如果没有实际数据，项目会自动使用虚拟数据集进行演示。

## 配置文件

项目配置文件位于 `src/configs/config.yaml`，包含以下主要配置项：

- `data`: 数据集相关配置
- `model`: 模型架构配置
- `train`: 训练参数配置
- `paths`: 路径配置
- `inference`: 推理配置

可以根据需要修改这些配置。

## 模型架构

项目实现了基于三阶段Mamba的神经网络架构：

1. **输入层**：接收224x224大小的RGB图像
2. **特征提取**：使用卷积层或分块嵌入提取视觉特征
3. **三阶段Mamba**：通过多个Mamba块组成的三阶段结构捕获长期依赖关系
4. **分类头**：将特征映射到10个驾驶员行为类别

## 训练模型

运行以下命令开始训练：

```bash
python src/train/train.py
```

训练过程中会：
- 在训练集上训练模型
- 在验证集上评估模型性能
- 保存最佳模型权重
- 生成训练历史图表
- 记录训练日志

## 模型评估

训练完成后，模型会自动在测试集上进行评估，并生成：
- 混淆矩阵（保存在results目录）
- 分类报告（包含准确率、精确率、召回率等指标）
- 测试集性能指标

## 推理预测

可以使用以下方式进行推理：

```python
from src.utils.inference import InferenceEngine

# 创建推理引擎
engine = InferenceEngine()

# 预测单个图像
results = engine.predict('path/to/image.jpg')
print(f"预测类别: {results['predicted_class_name']}")
print(f"置信度: {results['confidence']:.2%}")

# 可视化预测结果
engine.predict_with_visualization('path/to/image.jpg')

# 获取前3个预测结果
top_k = engine.get_top_k_predictions('path/to/image.jpg', k=3)
```

## 演示模式

如果没有实际数据集，可以直接运行训练脚本，系统会自动使用虚拟数据集进行演示：

```bash
python src/train/train.py
```

运行推理演示：

```bash
python src/utils/inference.py
```

## 结果解释

- **混淆矩阵**：展示模型在各类别上的分类情况，帮助识别容易混淆的类别
- **分类报告**：提供每个类别的精确率、召回率、F1分数等详细指标
- **训练历史**：展示训练过程中的损失和准确率变化，评估模型收敛情况

## 常见问题

1. **CUDA内存不足**：可以减小batch_size或图像大小
2. **训练不稳定**：尝试调整学习率或使用学习率调度器
3. **过拟合**：增加dropout率或使用数据增强

## 性能优化

- 使用GPU加速训练和推理
- 调整batch_size以充分利用GPU内存
- 考虑使用混合精度训练
- 针对部署场景，可以导出为ONNX格式

## 参考文献

- Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- State Farm Distracted Driver Detection Challenge

## 许可证

本项目采用MIT许可证。

## mamba安装过程的问题以及解决

参考网址：https://blog.csdn.net/yyywxk/article/details/145018635
解决在Win中安装Mamba的系列问题，包括不限于causal-conv1d的问题、mamba的安装问题等。
