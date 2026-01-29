import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import load_config, setup_matplotlib

# 加载配置文件
config_path = os.path.join('src', 'configs', 'config.yaml')
config = load_config(config_path)

# 从配置设置matplotlib，这将返回配置好的plt对象
plt = setup_matplotlib(config)

# 现在可以使用配置好的plt对象进行绘图
plt.figure(figsize=(10, 6))

# 绘制示例数据
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

plt.plot(x, y, 'b-o', linewidth=2, label='测试数据')

# 添加中文标签和标题（使用配置的字体）
plt.title('中文标题测试')
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.legend()

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 调整布局并保存（使用配置的DPI）
plt.tight_layout()
plt.savefig('example_plot.png', bbox_inches='tight')

print("图像已保存为 example_plot.png")
print("使用的matplotlib配置:")
print(f"- 后端: {config['matplotlib']['backend']}")
print(f"- 字体: {config['matplotlib']['font']['sans_serif']}")
print(f"- 图形DPI: {config['matplotlib']['figure_dpi']}")
print(f"- 保存DPI: {config['matplotlib']['savefig_dpi']}")