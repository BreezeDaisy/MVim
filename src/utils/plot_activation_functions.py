import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持（如果需要显示中文）
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义x范围
x = np.linspace(-5, 5, 1000)

# 定义激活函数
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 计算函数值
y_relu = relu(x)
y_sigmoid = sigmoid(x)

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制ReLU函数
plt.plot(x, y_relu, 'b-', linewidth=2, label='ReLU')

# 绘制Sigmoid函数
plt.plot(x, y_sigmoid, 'r-', linewidth=2, label='Sigmoid')

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 添加坐标轴标签
# plt.xlabel('输入值 x', fontsize=12)
# plt.ylabel('输出值', fontsize=12)

# 添加标题
plt.title('ReLU and Sigmoid functions', fontsize=14)

# 添加图例
plt.legend(fontsize=12)

# 设置坐标轴范围
plt.ylim(-0.1, 1.1)

# 添加关键点标记
# plt.scatter(0, 0, color='blue', s=50, zorder=5)
# plt.text(0.2, 0.05, '(0,0)', fontsize=10, color='blue')

# plt.scatter(0, 0.5, color='red', s=50, zorder=5)
# plt.text(0.2, 0.55, '(0,0.5)', fontsize=10, color='red')

# 设置刻度
plt.xticks(np.arange(-5, 6, 1))

# 保存图像
plt.tight_layout()
plt.savefig('activation_functions_comparison.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()