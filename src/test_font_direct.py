import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置字体渲染参数以提高清晰度
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['text.antialiased'] = True
matplotlib.rcParams['font.family'] = ['AR PL UMing CN']

# 启用抗锯齿和高质量文本渲染
plt.rcParams['text.antialiased'] = True
plt.rcParams['path.snap'] = False
plt.rcParams['image.interpolation'] = 'bilinear'
plt.rcParams['font.sans-serif'] = ['AR PL UMing CN']

# 设置全局字体大小
plt.rcParams['font.size'] = 14  # 基础字体大小
plt.rcParams['axes.titlesize'] = 20  # 标题字体大小
plt.rcParams['axes.labelsize'] = 16  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 14  # X轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 14  # Y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 14  # 图例字体大小

# 创建一个更高分辨率的图形
plt.figure(figsize=(10, 7), dpi=150)  # 增加DPI提高分辨率

# 绘制简单图形
plt.plot([1, 2, 3], [4, 5, 6], label='测试线', linewidth=2.5)  # 加粗线条
plt.title('测试中文显示') 
plt.xlabel('X轴')  
plt.ylabel('Y轴')  
plt.legend(['测试线'])

# 保存图形，使用更高的DPI
plt.tight_layout()
plt.savefig('test_chinese_font.png', dpi=300, bbox_inches='tight')  # 高DPI保存，确保文字清晰
print("图像已保存为 test_chinese_font.png")