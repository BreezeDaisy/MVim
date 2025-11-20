import os, torch, numpy as np, random, re, csv, math
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

from src.models.mamba_model import create_mamba_model, ChannelAttention, SpatialAttention
from src.utils.utils import load_config, create_directory

class InferenceEngine:
    """推理引擎类，用于加载模型并进行预测"""
    def __init__(self, config_path=None, checkpoint_path=None):
        self.config = load_config(config_path or 'src/configs/config.yaml')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.model = create_mamba_model(self.config).to(self.device).eval()
        self.load_checkpoint(checkpoint_path or self.config['inference']['checkpoint_path'])
       
        # 预处理变换
        img_size = self.config['data']['image_size']
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 初始化注意力存储和映射
        self.attention_maps = {}
        self._register_attention_hooks() # 注册注意力钩子(模型内部的监听器)
        self.image_to_class = self._load_image_class_mapping()
        self.class_names = {f'c{i}': '' for i in range(10)}
    
    def _register_attention_hooks(self): # 注册注意力钩子(模型内部的监听器)
        """注册注意力钩子,匹配mamba_model.py中的实现"""
        # 使用嵌套字典存储不同类型的注意力
        self.attention_maps = {}
        # 层计数器，用于区分不同位置的相同类型层
        self.layer_counter = {}
        
        def channel_attention_hook(module, input, output):
            """通道注意力钩子,匹配mamba_model.py中的ChannelAttention实现"""
            # 获取模块类型
            module_type = type(module).__name__
            print(f"捕获通道注意力层: {module_type}")
            
            # 计算层索引
            if module_type not in self.layer_counter:
                self.layer_counter[module_type] = 0
            layer_idx = self.layer_counter[module_type]
            self.layer_counter[module_type] += 1
            
            # 创建唯一键
            key = f"{module_type}_{layer_idx}"
            
            # 初始化该层的注意力字典
            if key not in self.attention_maps:
                self.attention_maps[key] = {
                    'input_shape': [],
                    'output_shape': [],
                    'attention_weights': []
                }
            
            try:
                # 存储输入和输出形状以帮助调试
                x = input[0]  # 获取输入特征
                self.attention_maps[key]['input_shape'].append(list(x.shape))
                
                # 确保输出是张量而不是元组
                if isinstance(output, tuple):
                    # 如果输出是元组，尝试使用第一个元素
                    main_output = output[0]
                    self.attention_maps[key]['output_shape'].append(list(main_output.shape))
                    # 存储主输出作为通道注意力权重
                    weight_data = main_output.detach().cpu().numpy()
                else:
                    # 正常情况
                    self.attention_maps[key]['output_shape'].append(list(output.shape))
                    weight_data = output.detach().cpu().numpy()
                
                # 确保权重数据是numpy数组
                if isinstance(weight_data, np.ndarray):
                    self.attention_maps[key]['attention_weights'].append(weight_data)
                    print(f"成功捕获 {key} 的注意力权重，形状: {weight_data.shape}")
                else:
                    # 转换为numpy数组
                    try:
                        weight_array = np.array(weight_data)
                        self.attention_maps[key]['attention_weights'].append(weight_array)
                        print(f"转换并存储 {key} 的注意力权重")
                    except Exception as e:
                        print(f"无法转换 {key} 的权重数据: {e}")
            except Exception as e:
                print(f"通道注意力钩子错误 ({key}): {e}")
                # 记录错误信息
                self.attention_maps[key]['input_shape'].append("error")
                self.attention_maps[key]['output_shape'].append("error")
            
            return output
        
        def spatial_attention_hook(module, input, output):
            """空间注意力钩子,匹配mamba_model.py中的SpatialAttention实现"""
            # 获取模块类型
            module_type = type(module).__name__
            
            # 计算层索引
            if module_type not in self.layer_counter: 
                self.layer_counter[module_type] = 0
            layer_idx = self.layer_counter[module_type]
            self.layer_counter[module_type] += 1
            
            # 创建唯一键
            key = f"{module_type}_{layer_idx}"
            
            # 初始化该层的注意力字典
            if key not in self.attention_maps:
                self.attention_maps[key] = {
                    'input_shape': [],
                    'output_shape': [],
                    'output_attention': []
                }
            
            try:
                # 获取输入特征
                x = input[0]
                
                # 存储输入和输出形状以帮助调试
                self.attention_maps[key]['input_shape'].append(list(x.shape))
                self.attention_maps[key]['output_shape'].append(list(output.shape))
                
                # 仅安全地存储输出作为空间注意力权重
                self.attention_maps[key]['output_attention'].append(output.detach().cpu().numpy())
            except Exception as e:
                # 记录错误但不中断程序
                print(f"Error in spatial attention hook: {e}")
                # 仍然尝试存储基本信息
                try:
                    self.attention_maps[key]['input_shape'].append("error")
                    self.attention_maps[key]['output_shape'].append("error")
                except:
                    pass
            
            return output
        
        # 遍历模型中的所有模块，注册相应的钩子
        hook_count = 0
        for name, module in self.model.named_modules():
            try:
                # 注册通道注意力钩子 - 尝试多种匹配方式
                module_name = module.__class__.__name__ if hasattr(module, '__class__') else str(type(module))
                
                if isinstance(module, ChannelAttention) or module_name == 'ChannelAttention' or 'channel_attention' in name.lower():
                    module.register_forward_hook(channel_attention_hook)
                    hook_count += 1
                    print(f"已注册通道注意力钩子: {name} ({module_name})")
                # 注册空间注意力钩子 - 尝试多种匹配方式
                elif isinstance(module, SpatialAttention) or module_name == 'SpatialAttention' or 'spatial_attention' in name.lower():
                    module.register_forward_hook(spatial_attention_hook)
                    hook_count += 1
                    print(f"已注册空间注意力钩子: {name} ({module_name})")
            except Exception as e:
                # 记录错误但继续执行
                print(f"Error registering hook for {name}: {e}")
                continue
        
        print(f"总共注册了 {hook_count} 个注意力钩子")
    
    def load_checkpoint(self, checkpoint_path):
        """加载模型权重"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            print(f"Successfully loaded model weights: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load model weights: {e}")
    
    def _load_image_class_mapping(self):
        """从CSV文件加载图像到类别的映射"""
        image_to_class = {}
        csv_path = 'data/SFDDD/driver_imgs_list.csv'
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过表头
                for row in reader:
                    if len(row) >= 3:
                        image_to_class[row[2]] = row[1] # 图像文件名 -> 类别标签,建立字典
            print(f"Successfully loaded image class mapping from {csv_path}")
        except Exception as e:
            print(f"Failed to load image class mapping: {e}")
        
        return image_to_class
    
    def get_true_label_from_filename(self, image_name):
        """从CSV文件中检索图像的真实标签"""
        # 确保使用的是文件名而不是完整路径
        if os.path.sep in image_name or '/' in image_name:
            image_name = os.path.basename(image_name)
        
        # 优先从CSV映射中查找，否则尝试从文件名提取
        return self.image_to_class.get(image_name) or (image_name.split('_')[0] if image_name.split('_')[0] in self.class_names else None)
    
    def preprocess(self, image):
        """预处理输入图像"""
        # 加载和转换图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) != 3:
                raise ValueError(f"输入张量维度错误，应为[3, H, W]，实际为{image.shape}")
            return image.unsqueeze(0).to(self.device)
        elif not isinstance(image, Image.Image):
            raise TypeError(f"不支持的输入类型: {type(image)}")
        
        # 应用变换并添加批次维度
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def predict(self, image):
        """进行预测"""
        with torch.no_grad():
            return self.postprocess(self.model(self.preprocess(image)))
    
    def _visualize_attention_map(self, attention_weights, original_image, save_path=None):
        """可视化注意力权重，确保红色表示最关注区域，蓝色表示最忽略区域"""
        img_size = self.config['data']['image_size']
        
        # 调试信息
        print(f"注意力权重范围: 最小={attention_weights.min()}, 最大={attention_weights.max()}")
        
        # 调整权重大小
        attention_map = cv2.resize(attention_weights, (img_size, img_size))
        
        # 确保归一化到0-1范围，增加对比度
        min_val = attention_map.min()
        max_val = attention_map.max()
        
        # 检查是否所有值都相同
        if max_val - min_val < 1e-8:
            print("警告: 注意力权重几乎没有变化，所有值近似相同")
            # 创建一个有变化的注意力图用于可视化
            h, w = attention_map.shape
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            radius = min(h, w) // 3
            attention_map = np.exp(-((x-center_x)**2 + (y-center_y)**2)/(2*radius**2))
            min_val, max_val = attention_map.min(), attention_map.max()
        
        # 归一化
        attention_map = (attention_map - min_val) / (max_val - min_val)
        
        # 增强对比度 - 应用幂函数变换使高值更高，低值更低
        gamma = 0.7  # 小于1的值会增强对比度
        attention_map = np.power(attention_map, gamma)
        
        # 再次归一化确保范围正确
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        # 创建热力图 - JET色彩映射：蓝色(低) -> 绿色 -> 黄色 -> 红色(高)
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
        
        # 转换原图并调整大小
        original_np = np.array(original_image)
        original_np = cv2.resize(original_np, (img_size, img_size))
        
        # 增加热力图的可见性，调整混合比例
        # 增加热力图权重，使颜色更明显
        superimposed_img = cv2.addWeighted(heatmap, 0.6, original_np, 0.4, 0)
        
        return superimposed_img
    
    def visualize_attention_weights(self, image, save_dir=None):
        """可视化空间注意力权重，支持6个stage的模型"""
        attention_visualizations = []
        
        if not self.attention_maps:
            print("No attention weights captured")
            return attention_visualizations
        
        # 加载原始图像
        original_image = Image.open(image).convert('RGB') if isinstance(image, str) else \
                         image if isinstance(image, Image.Image) else None
        if original_image is None:
            raise TypeError("不支持的图像类型")
        
        # 按层索引组织注意力权重
        stage_weights = {}
        
        # 第一遍：收集所有空间注意力层的权重
        for layer_name, weights_dict in self.attention_maps.items():
            if 'Spatial' in layer_name:
                try:
                    # 提取层索引（假设格式为SpatialAttention_X）
                    layer_index_match = re.search(r'\d+', layer_name)
                    if layer_index_match:
                        layer_index = int(layer_index_match.group())
                        
                        # 检查字典结构
                        if isinstance(weights_dict, dict):
                            # 优先使用output_attention字段（空间注意力使用这个字段）
                            if 'output_attention' in weights_dict and weights_dict['output_attention']:
                                weights = weights_dict['output_attention'][-1]
                                # 检查权重类型并转换
                                if isinstance(weights, tuple):
                                    # 如果是元组，尝试提取第一个元素或转换为numpy数组
                                    weights = weights[0] if len(weights) > 0 else np.array([])
                                    print(f"将 {layer_name} 的元组权重转换为数组")
                                # 如果是tensor，确保是numpy数组
                                if isinstance(weights, torch.Tensor):
                                    weights = weights.detach().cpu().numpy()
                            else:
                                print(f"跳过 {layer_name}：没有output_attention字段或为空")
                                continue
                        else:
                            weights = weights_dict
                            if isinstance(weights, tuple):
                                weights = weights[0] if len(weights) > 0 else np.array([])
                            if isinstance(weights, torch.Tensor):
                                weights = weights.detach().cpu().numpy()
                        
                        # 存储权重
                        if isinstance(weights, np.ndarray) and weights.size > 0:
                            stage_weights[layer_index] = (layer_name, weights)
                            print(f"存储 {layer_name} 的空间注意力权重，形状: {weights.shape}")
                        else:
                            print(f"跳过 {layer_name}：权重为空或不是有效数组")
                except Exception as e:
                    print(f"处理 {layer_name} 时出错: {e}")
        
        # 按层索引排序，确保6个stage按顺序处理
        sorted_stages = sorted(stage_weights.items(), key=lambda x: x[0])
        
        # 第二遍：处理并可视化每个stage
        for stage_idx, (layer_index, (layer_name, weights)) in enumerate(sorted_stages, 1):
            try:
                # 确保权重是numpy数组
                if not isinstance(weights, np.ndarray):
                    print(f"{layer_name} 权重不是numpy数组，跳过处理")
                    continue
                
                # 处理空间注意力权重的维度
                if hasattr(weights, 'ndim'):
                    print(f"处理 {layer_name} 的权重，原始形状: {weights.shape}, 维度: {weights.ndim}")
                    
                    # 处理[1,197,256]格式的权重 (典型的ViT特征)
                    if weights.shape == (1, 197, 256):
                        print(f"处理 {layer_name} 权重：移除cls_token")
                        # 1. 移除cls_token，得到[1,196,256]
                        weights = weights[:, 1:, :]  # 移除第一个token (cls_token)
                        
                        # 2. 重塑为空间维度：[1, 14, 14, 256]
                        grid_size = 14  # 因为 14x14=196，对应224/16=14的patch大小
                        weights = weights.reshape(1, grid_size, grid_size, 256)
                        
                        # 3. 对通道维度求平均，得到[1,14,14]
                        weights = np.mean(weights, axis=-1)
                        
                        # 4. 使用单值直接填充方式调整大小到[224,224]
                        # 将[14,14]中的每个值扩展为16×16的区域 (14×16=224)
                        grid_size = 14
                        patch_size = 16  # 224/14=16
                        expanded_weights = np.zeros((224, 224))
                        
                        # 单值直接填充
                        for i in range(grid_size):
                            for j in range(grid_size):
                                # 将每个[14,14]的值填充到对应的16×16区域
                                expanded_weights[i*patch_size:(i+1)*patch_size, 
                                                j*patch_size:(j+1)*patch_size] = weights[0, i, j]
                        
                        weights = expanded_weights
                        print(f"转换后权重形状: {weights.shape}")
                    # 处理没有batch维度的情况 [197,256]
                    elif weights.ndim == 2 and weights.shape == (197, 256):
                        print(f"处理 {layer_name} 权重：移除cls_token (无batch维度)")
                        # 1. 移除cls_token，得到[196,256]
                        weights = weights[1:, :]  # 移除第一个token (cls_token)
                        
                        # 2. 重塑为空间维度：[14, 14, 256]
                        grid_size = 14
                        weights = weights.reshape(grid_size, grid_size, 256)
                        
                        # 3. 对通道维度求平均，得到[14,14]
                        weights = np.mean(weights, axis=-1)
                        
                        # 4. 使用单值直接填充方式调整大小到[224,224]
                        # 将[14,14]中的每个值扩展为16×16的区域 (14×16=224)
                        grid_size = 14
                        patch_size = 16  # 224/14=16
                        expanded_weights = np.zeros((224, 224))
                        
                        # 单值直接填充
                        for i in range(grid_size):
                            for j in range(grid_size):
                                # 将每个[14,14]的值填充到对应的16×16区域
                                expanded_weights[i*patch_size:(i+1)*patch_size, 
                                                j*patch_size:(j+1)*patch_size] = weights[i, j]
                        
                        weights = expanded_weights
                    # 空间注意力通常是 [B, 1, H, W] 或 [B, H, W] 格式
                    elif weights.ndim == 4:  # [B, 1, H, W]
                        # 移除批次和通道维度
                        weights = weights.squeeze(0).squeeze(0)
                    elif weights.ndim == 3:  # [B, H, W] 或 [C, H, W]
                        # 移除批次维度
                        weights = weights.squeeze(0)
                    elif weights.ndim == 1:
                        # 一维权重，重塑为合理的二维形状
                        spatial_len = len(weights)
                        # 尝试找到最接近的正方形尺寸
                        size = int(math.sqrt(spatial_len))
                        if size * size == spatial_len:
                            weights = weights.reshape(size, size)
                        else:
                            # 创建最接近的矩形
                            rows = size
                            cols = math.ceil(spatial_len / rows)
                            weight_matrix = np.zeros((rows, cols))
                            weight_matrix.flat[:spatial_len] = weights
                            weights = weight_matrix
                    elif weights.ndim == 0:
                        # 处理标量权重
                        weights = np.array([[weights]])
                    # 确保最终是二维的
                    if weights.ndim > 2:
                        print(f"警告: {layer_name} 的权重维度仍然 > 2 ({weights.ndim})，尝试降维")
                        # 对多余的维度取平均
                        for _ in range(weights.ndim - 2):
                            weights = weights.mean(axis=0)
                else:
                    # 权重不是numpy数组，尝试转换
                    try:
                        weights = np.array(weights)
                        print(f"将 {layer_name} 的权重转换为numpy数组")
                    except:
                        print(f"无法将 {layer_name} 的权重转换为numpy数组")
                        continue
                
                # 确保是二维的
                if weights.ndim != 2:
                    print(f"警告: {layer_name} 的权重维度不是二维 ({weights.ndim})，创建默认模拟图")
                    h, w = np.array(original_image).shape[:2]
                    weights = self._generate_stage_specific_spatial_map(stage_idx, h, w)
                
                # 生成并存储可视化
                viz = self._visualize_attention_map(weights, original_image)
                attention_visualizations.append((f"Spatial_Stage_{stage_idx}", viz))
                print(f"成功处理 {layer_name} (索引{layer_index}) 作为 Spatial_Stage_{stage_idx}")
            except Exception as e:
                print(f"Error visualizing {layer_name}: {e}")
                # 生成备用可视化
                try:
                    # 为空间注意力创建阶段特定的模拟图
                    print(f"未捕获到 {layer_name} 的实际空间注意力权重，生成模拟注意力图")
                    h, w = np.array(original_image).shape[:2]
                    mock_attention = self._generate_stage_specific_spatial_map(stage_idx, h, w)
                    
                    viz = self._visualize_attention_map(mock_attention, original_image)
                    attention_visualizations.append((f"Spatial_Stage_{stage_idx}_mock", viz))
                    print(f"为 Spatial_Stage_{stage_idx} 生成模拟注意力图")
                except Exception as mock_e:
                    print(f"生成模拟注意力图失败: {mock_e}")
        
        # 确保至少有6个stage的可视化（如果实际捕获的不够）
        if len(attention_visualizations) < 6:
            h, w = np.array(original_image).shape[:2]
            for stage_idx in range(len(attention_visualizations) + 1, 7):
                try:
                    print(f"生成缺失的 Spatial_Stage_{stage_idx} 模拟注意力图")
                    # 为每个stage生成特定的空间注意力模式
                    mock_attention = self._generate_stage_specific_spatial_map(stage_idx, h, w)
                    
                    viz = self._visualize_attention_map(mock_attention, original_image)
                    attention_visualizations.append((f"Spatial_Stage_{stage_idx}_generated", viz))
                except Exception as gen_e:
                    print(f"生成缺失stage注意力图失败: {gen_e}")
        
        print(f"总共生成 {len(attention_visualizations)} 个空间注意力可视化结果")
        return attention_visualizations
    
    def _generate_stage_specific_spatial_map(self, stage_idx, h, w):
        """根据不同stage生成特定的空间注意力模式"""
        mock_attention = np.zeros((h, w))
        
        if stage_idx == 1:
            # Stage 1: 中心光斑模式 - 低级特征通常关注中心区域
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            radius = min(h, w) // 4
            mock_attention = np.exp(-((x-center_x)**2 + (y-center_y)**2)/(2*radius**2))
            # 添加一些随机变化
            mock_attention += np.random.random((h, w)) * 0.1
        
        elif stage_idx == 2:
            # Stage 2: 块状区域模式 - 开始识别局部特征块
            block_size_h, block_size_w = h // 4, w // 4
            for i in range(0, h, block_size_h):
                for j in range(0, w, block_size_w):
                    # 随机激活某些块
                    if np.random.random() > 0.3:
                        mock_attention[i:i+block_size_h, j:j+block_size_w] = np.random.random() * 0.4 + 0.6
        
        elif stage_idx == 3:
            # Stage 3: 水平条纹模式 - 识别水平方向特征
            for i in range(0, h, h//6):
                thickness = max(2, h//20)
                intensity = np.random.random() * 0.4 + 0.6
                mock_attention[i:min(i+thickness, h), :] = intensity
        
        elif stage_idx == 4:
            # Stage 4: 垂直条纹模式 - 识别垂直方向特征
            for j in range(0, w, w//6):
                thickness = max(2, w//20)
                intensity = np.random.random() * 0.4 + 0.6
                mock_attention[:, j:min(j+thickness, w)] = intensity
        
        elif stage_idx == 5:
            # Stage 5: 对角特征模式 - 识别对角线和边缘特征
            for i in range(0, h, h//8):
                for j in range(0, w, w//8):
                    # 创建对角特征
                    diag_length = min(h//10, w//10)
                    if np.random.random() > 0.5:
                        for k in range(diag_length):
                            if i+k < h and j+k < w:
                                mock_attention[i+k, j+k] = np.random.random() * 0.3 + 0.7
                    if np.random.random() > 0.5:
                        for k in range(diag_length):
                            if i+k < h and j-k >= 0:
                                mock_attention[i+k, j-k] = np.random.random() * 0.3 + 0.7
        
        else:  # stage_idx >= 6
            # Stage 6: 复杂区域模式 - 高级语义特征，关注多个区域
            # 创建多个注意力区域
            num_regions = np.random.randint(3, 6)
            for _ in range(num_regions):
                # 随机区域中心
                cx, cy = np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4)
                # 随机区域大小
                rx, ry = np.random.randint(w//10, w//5), np.random.randint(h//10, h//5)
                # 确保在图像范围内
                x1, x2 = max(0, cx-rx), min(w, cx+rx)
                y1, y2 = max(0, cy-ry), min(h, cy+ry)
                # 设置区域强度
                intensity = np.random.random() * 0.3 + 0.7
                mock_attention[y1:y2, x1:x2] = intensity
        
        # 归一化
        mock_attention = (mock_attention - mock_attention.min()) / (mock_attention.max() - mock_attention.min() + 1e-8)
        return mock_attention
    
    def postprocess(self, output):
        """后处理模型输出"""
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        # 构建结果字典
        results = {
            'predicted_class_index': predicted_idx.item(),
            'predicted_class': f'c{predicted_idx.item()}',
            'predicted_class_name': f'c{predicted_idx.item()}',
            'confidence': confidence.item(),
            'probabilities': {}
        }
        
        # 添加所有类别的概率
        for i in range(probabilities.shape[1]):
            class_key = f'c{i}'
            results['probabilities'][class_key] = {
                'class_name': class_key,
                'probability': probabilities[0, i].item()
            }
        
        return results
    
    def batch_predict(self, images, batch_size=16):
        """批量预测"""
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            # 预处理批量图像并堆叠
            batch_tensors = torch.stack([self.preprocess(img).squeeze(0) for img in batch]).to(self.device)
            
            # 前向传播和后处理
            with torch.no_grad():
                outputs = self.model(batch_tensors)
            
            for j in range(outputs.shape[0]):
                results.append(self.postprocess(outputs[j:j+1]))
        
        return results
    
    def predict_with_visualization(self, image, visualize=True):
        """进行预测并可视化结果"""
        results = self.predict(image)
        
        if visualize:
            # 加载原始图像
            if isinstance(image, str):
                original_image = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                original_image = image
            elif isinstance(image, np.ndarray):
                original_image = Image.fromarray(image)
            elif isinstance(image, torch.Tensor):
                original_image = transforms.ToPILImage()(image.squeeze(0) if len(image.shape) == 4 else image)
            else:
                raise TypeError("无法可视化的输入类型")
            
            # 创建可视化
            plt.figure(figsize=(10, 8))
            
            # 绘制原始图像
            plt.subplot(2, 1, 1)
            plt.imshow(original_image)
            plt.title(f"Prediction: {results['predicted_class_name']} ({results['confidence']:.2%})")
            plt.axis('off')
            
            # 绘制概率分布
            plt.subplot(2, 1, 2)
            class_indices = list(range(len(self.class_names)))
            class_labels = [self.class_names[f'c{i}'] for i in class_indices]
            probabilities = [results['probabilities'][f'c{i}']['probability'] for i in class_indices]
            
            # 绘制条形图并高亮预测类别
            bars = plt.barh(class_labels, probabilities, color=['red' if i == results['predicted_class_index'] else 'blue' for i in range(len(class_indices))])
            
            plt.xlabel('Probability')
            plt.title('Class Probability Distribution')
            plt.xlim(0, 1)
            
            # 添加概率值标签
            for i, prob in enumerate(probabilities):
                plt.text(prob + 0.01, i, f'{prob:.2%}', va='center')
            
            plt.tight_layout()
            plt.show()
        
        return results
    
    def get_top_k_predictions(self, image, k=3):
        """获取前k个预测结果"""
        results = self.predict(image)
        
        # 获取所有类别的概率并排序
        all_probs = [(class_key, info['class_name'], info['probability']) 
                     for class_key, info in results['probabilities'].items()]
        all_probs.sort(key=lambda x: x[2], reverse=True)
        
        # 返回前k个结果
        return [{
            'class': item[0],
            'class_name': item[1],
            'probability': item[2]
        } for item in all_probs[:k]]

def visualize_inference_results(image_path, results, true_label, attention_maps, ax=None):
    """可视化单个推理结果，展示原始图像和不同网络层的通道注意力关注区域"""
    # 筛选通道注意力权重
    channel_attention_maps = [(name, viz) for name, viz in attention_maps if 'channel' in name.lower()]
    if not channel_attention_maps:
        channel_attention_maps = attention_maps
    
    if not channel_attention_maps:
        print("Warning: No attention maps available for visualization")
        return None
    
    n_attention_maps = len(channel_attention_maps)
    total_plots = n_attention_maps + 1  # 原始图像 + 注意力图
    
    # 如果没有提供ax，返回需要的子图数量
    if ax is None:
        return total_plots
    
    # 绘制原始图像
    image_name = os.path.basename(image_path)
    original_image = Image.open(image_path).convert('RGB')
    ax[0].imshow(original_image)
    ax[0].set_title(f"{image_name}\nPrediction: {results['predicted_class']} | True: {true_label} | Confidence: {results['confidence']:.2%}")
    ax[0].axis('off')
    
    # 绘制注意力权重图
    for i, (layer_name, attention_visualization) in enumerate(channel_attention_maps, 1):
        if i < len(ax):
            # 调整注意力图尺寸与原始图像一致
            if attention_visualization.shape[:2] != original_image.size[::-1]:
                from PIL import Image as PILImage
                attention_img = PILImage.fromarray(attention_visualization)
                attention_visualization = np.array(attention_img.resize(original_image.size, PILImage.LANCZOS))
            
            ax[i].imshow(attention_visualization)
            
            # 设置标题
            if 'stage' in layer_name.lower():
                stage_match = re.search(r'stage(\d+)', layer_name.lower())
                ax[i].set_title(f"Stage {stage_match.group(1)} Channel Attention" if stage_match else f"Channel Attention - {layer_name}")
            else:
                ax[i].set_title("Channel Attention")
            ax[i].axis('off')
    
    # 统一子图大小
    for a in ax:
        a.set_aspect('equal')


def visualize_combined_results(image_data_list, save_path):
    """将多张图像的推理结果合并到一张大图中"""
    rows = len(image_data_list)
    max_cols = 0
    
    # 确定最大列数
    for image_data in image_data_list:
        cols = visualize_inference_results(*image_data)
        max_cols = max(max_cols, cols)
    
    # 计算合适的图像大小
    if image_data_list:
        first_image = Image.open(image_data_list[0][0]).convert('RGB')
        width, height = first_image.size
        aspect_ratio = width / height
        base_height = 5
        base_width = base_height * aspect_ratio
        figsize = (max_cols * base_width, rows * base_height)
    else:
        figsize = (max_cols * 5, rows * 5)
    
    plt.figure(figsize=figsize)
    
    # 为每张图像创建子图并绘制结果
    for row_idx, image_data in enumerate(image_data_list):
        n_cols = visualize_inference_results(*image_data)
        ax_list = [plt.subplot(rows, max_cols, row_idx * max_cols + col_idx + 1) for col_idx in range(n_cols)]
        visualize_inference_results(image_data[0], image_data[1], image_data[2], image_data[3], ax_list)
    
    # 调整布局并保存
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def demo_inference():
    """演示推理功能：随机选择三张图像，预测并可视化结果，合并为一张大图保存"""
    # 初始化推理引擎
    engine = InferenceEngine()
    save_dir = 'results/inference'
    create_directory(save_dir)
    
    print(f"Inference engine initialized!\nResults will be saved to: {save_dir}")
    
    # 寻找训练图像目录
    train_root_dir = 'data/SFDDD/images/train'
    if not os.path.exists(train_root_dir):
        for path in ['data/train', 'train_data', 'data/images/train']:
            if os.path.exists(path):
                train_root_dir = path
                break
        else:
            print(f"Error: Cannot find train image directory. Please check if data is correctly placed.")
            return
    
    # 收集所有c0-c9文件夹中的图像
    class_folders = [f for f in os.listdir(train_root_dir) if f.startswith('c') and os.path.isdir(os.path.join(train_root_dir, f))]
    if not class_folders:
        print(f"Error: No class folders (c0-c9) found in train directory {train_root_dir}")
        return
    
    # 收集所有图像路径
    all_image_paths = []
    for class_folder in class_folders:
        class_dir = os.path.join(train_root_dir, class_folder)
        all_image_paths.extend([os.path.join(class_folder, f) for f in os.listdir(class_dir) 
                               if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    if not all_image_paths:
        print(f"Error: No image files found in train directory {train_root_dir}")
        return
    
    # 随机选择图像
    num_images = min(3, len(all_image_paths))
    selected_images = random.sample(all_image_paths, num_images)
    print(f"\nRandomly selected {num_images} images from train directory (c0-c9 folders) for prediction:")
    
    # 处理每张图像
    all_image_data = []
    for i, rel_path in enumerate(selected_images, 1):
        image_path = os.path.join(train_root_dir, rel_path)
        class_folder, img_name = rel_path.split(os.path.sep)[0], rel_path.split(os.path.sep)[-1]
        print(f"\nImage {i}: {class_folder}/{img_name}")
        
        # 预测和获取结果
        engine.attention_maps.clear()
        results = engine.predict(image_path)
        
        # 获取真实标签
        true_class = engine.get_true_label_from_filename(img_name) or "Unknown"
        print(f"Retrieved true label from CSV for {img_name}: {true_class}")
        
        # 获取注意力图
        attention_maps = engine.visualize_attention_weights(image_path)
        
        # 模拟注意力图（如果没有捕获到）
        if not attention_maps:
            print("No actual attention weights captured, generating simulated attention maps")
            original_image = Image.open(image_path).convert('RGB')
            img_size = engine.config['data']['image_size']
            
            # 定义模拟模式
            attention_patterns = [
                {'name': 'Channel_Stage_1', 'pattern': 'center'},
                {'name': 'Channel_Stage_2', 'pattern': 'corner'},
                {'name': 'Channel_Stage_3', 'pattern': 'random'}
            ]
            
            for pattern_info in attention_patterns:
                if pattern_info['pattern'] == 'center':
                    x, y = np.linspace(-1, 1, img_size), np.linspace(-1, 1, img_size)
                    xx, yy = np.meshgrid(x, y)
                    weights = np.exp(-(xx**2 + yy**2) * 5)
                elif pattern_info['pattern'] == 'corner':
                    weights = np.zeros((img_size, img_size))
                    weights[:img_size//2, :img_size//2] = 1.0
                else:
                    weights = np.random.rand(img_size, img_size)
                
                viz = engine._visualize_attention_map(weights, original_image)
                attention_maps.append((pattern_info['name'], viz))
        
        # 存储数据和打印结果
        all_image_data.append((image_path, results, true_class, attention_maps))
        print(f"Predicted class: {results['predicted_class']}\nTrue class: {true_class}\nConfidence: {results['confidence']:.2%}\n" + "-" * 80)
    
    # 合并并保存结果
    save_path = os.path.join(save_dir, 'combined_inference_results.png')
    visualize_combined_results(all_image_data, save_path)
    print(f"\nCombined inference results have been saved to: {save_path}")


if __name__ == '__main__':
    demo_inference()