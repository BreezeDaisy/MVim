import os

def count_images_by_prefix(directory, prefix):
    """
    统计目录中以指定前缀开头的.jpg文件数量
    Args:
        directory: 目录路径
        prefix: 文件前缀
    Returns:
        int: 文件数量
    """
    count = 0
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith('.jpg'):
            count += 1
    return count

def main():
    # 定义Emotion_dataset目录路径
    emotion_dataset_dir = "Face_Pose/Emotion_dataset"
    
    # 检查目录是否存在
    if not os.path.exists(emotion_dataset_dir):
        print(f"错误: 目录 {emotion_dataset_dir} 不存在")
        return
    
    # 遍历所有子目录
    print("统计Emotion_dataset目录下各子目录的图片数量:")
    print("=" * 60)
    
    # 存储总统计结果
    total_train = 0
    total_validation = 0
    
    for subdir in sorted(os.listdir(emotion_dataset_dir)):
        subdir_path = os.path.join(emotion_dataset_dir, subdir)
        
        # 确保是目录
        if not os.path.isdir(subdir_path):
            continue
        
        # 统计train开头的图片数量
        train_count = count_images_by_prefix(subdir_path, "train_")
        
        # 统计validation开头的图片数量
        validation_count = count_images_by_prefix(subdir_path, "validation_")
        
        # 累计总数
        total_train += train_count
        total_validation += validation_count
        
        # 输出结果
        print(f"目录: {subdir}")
        print(f"  train开头: {train_count}")
        print(f"  validation开头: {validation_count}")
        print(f"  总计: {train_count + validation_count}")
        print("-" * 40)
    
    # 输出总计
    print("总计:")
    print(f"train开头图片总数: {total_train}")
    print(f"validation开头图片总数: {total_validation}")
    print(f"所有图片总数: {total_train + total_validation}")



if __name__ == "__main__":
    main()