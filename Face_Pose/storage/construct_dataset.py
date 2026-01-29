import os
import random
import shutil
from tqdm import tqdm

# ======================================================
# 配置参数
# ======================================================

# 原始数据集路径
ORIGINAL_DATASET_DIR = "Face_Pose/Emotion_dataset"

# 新数据集输出路径
NEW_DATASET_DIR = "Face_Pose/Constructed_Emotion_Dataset"

# 抽取比例
EXTRACTION_RATIOS = {
    "happy": 0.25,    # 1/4
    "neutral": 1/3,    # 1/3
    "sad": 0.5         # 1/2
}

# 训练集和验证集的划分比例
TRAIN_VAL_SPLIT = 0.8  # 80% 训练集，20% 验证集

# 情绪类别映射
EMOTION_MAPPING = {
    "angry": "antipathic",
    "disgust": "antipathic",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprise": "surprise"
}

# ======================================================
# 工具函数
# ======================================================

def get_all_jpg_files(directory):
    """
    获取目录下所有.jpg文件
    
    Args:
        directory: 目录路径
    
    Returns:
        list: .jpg文件路径列表
    """
    jpg_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            jpg_files.append(os.path.join(directory, filename))
    return jpg_files

def random_sample_files(file_list, ratio):
    """
    随机抽取指定比例的文件
    
    Args:
        file_list: 文件列表
        ratio: 抽取比例
    
    Returns:
        list: 抽取的文件列表
    """
    sample_size = int(len(file_list) * ratio)
    return random.sample(file_list, sample_size)

def split_train_val(file_list, train_ratio):
    """
    将文件列表划分为训练集和验证集
    
    Args:
        file_list: 文件列表
        train_ratio: 训练集比例
    
    Returns:
        tuple: (训练集文件列表, 验证集文件列表)
    """
    random.shuffle(file_list)
    split_idx = int(len(file_list) * train_ratio)
    return file_list[:split_idx], file_list[split_idx:]

def copy_files_with_new_name(file_list, destination_dir, prefix, start_idx=0):
    """
    复制文件到目标目录，并使用新的命名格式
    Args:
        file_list: 源文件列表
        destination_dir: 目标目录
        prefix: 文件前缀 (train_ 或 val_)
        start_idx: 起始索引
    Returns:
        int: 复制的文件数量
    """
    os.makedirs(destination_dir, exist_ok=True)
    
    count = 0
    for i, src_file in enumerate(tqdm(file_list, desc=f"复制到 {destination_dir}")):
        new_filename = f"{prefix}{start_idx + i}.jpg"
        dst_file = os.path.join(destination_dir, new_filename)
        shutil.copy2(src_file, dst_file)
        count += 1
    
    return count

# ======================================================
# 主函数
# ======================================================

def main():
    # 确保原始数据集存在
    if not os.path.exists(ORIGINAL_DATASET_DIR):
        print(f"错误: 原始数据集目录 {ORIGINAL_DATASET_DIR} 不存在")
        return
    
    # 清空并创建新数据集目录
    if os.path.exists(NEW_DATASET_DIR):
        shutil.rmtree(NEW_DATASET_DIR)
    os.makedirs(NEW_DATASET_DIR, exist_ok=True)
    
    print("开始构造数据集...")
    
    # 第一步：收集和处理各类别文件
    emotion_files = {}
    
    # 遍历原始数据集的所有子目录
    for emotion_dir in os.listdir(ORIGINAL_DATASET_DIR):
        emotion_path = os.path.join(ORIGINAL_DATASET_DIR, emotion_dir)
        
        if not os.path.isdir(emotion_path):
            continue
        
        # 获取当前情绪目录下的所有.jpg文件
        jpg_files = get_all_jpg_files(emotion_path)
        
        # 映射到新的情绪类别
        new_emotion = EMOTION_MAPPING.get(emotion_dir, emotion_dir)
        
        # 应用抽取比例
        if emotion_dir in EXTRACTION_RATIOS:
            ratio = EXTRACTION_RATIOS[emotion_dir]
            jpg_files = random_sample_files(jpg_files, ratio)
            print(f"从 {emotion_dir} 中抽取 {ratio:.2f} 的文件，共 {len(jpg_files)} 个")
        
        # 合并到新的情绪类别
        if new_emotion not in emotion_files:
            emotion_files[new_emotion] = []
        emotion_files[new_emotion].extend(jpg_files)
    
    # 第二步：划分训练集和验证集
    train_files = {}
    val_files = {}
    
    print("\n划分训练集和验证集...")
    for emotion, files in emotion_files.items():
        train, val = split_train_val(files, TRAIN_VAL_SPLIT)
        train_files[emotion] = train
        val_files[emotion] = val
        print(f"{emotion}: 训练集 {len(train)} 个，验证集 {len(val)} 个")
    
    # 第三步：复制文件到新目录并统一命名
    print("\n复制文件到新目录并统一命名...")
    
    # 复制训练集
    train_base_dir = os.path.join(NEW_DATASET_DIR, "train")
    train_idx = 0
    
    for emotion, files in train_files.items():
        emotion_train_dir = os.path.join(train_base_dir, emotion)
        copied = copy_files_with_new_name(files, emotion_train_dir, "train_", train_idx)
        train_idx += copied
        print(f"复制 {emotion} 训练集: {copied} 个文件")
    
    # 复制验证集
    val_base_dir = os.path.join(NEW_DATASET_DIR, "val")
    val_idx = 0
    
    for emotion, files in val_files.items():
        emotion_val_dir = os.path.join(val_base_dir, emotion)
        copied = copy_files_with_new_name(files, emotion_val_dir, "val_", val_idx)
        val_idx += copied
        print(f"复制 {emotion} 验证集: {copied} 个文件")
    
    print("\n数据集构造完成！")
    print(f"新数据集保存路径: {NEW_DATASET_DIR}")

if __name__ == "__main__":
    # 设置随机种子，确保结果可重现
    random.seed(42)
    main()