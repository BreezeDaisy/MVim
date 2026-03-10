import os

def count_images(directory):
    """
    统计目录中所有.jpg文件数量
    Args:
        directory: 目录路径
    Returns:
        int: 文件数量
        set: 文件名集合
    """
    count = 0
    filenames = set()
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            count += 1
            filenames.add(filename)
    return count, filenames

def main():
    # 定义small_data目录路径
    data_dir = "/home/zdx/python_daima/MVim/MVim/Face_Pose/small_data"
    
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误: 目录 {data_dir} 不存在")
        return
    
    print("统计small_data目录下各子目录的图片数量:")
    print("=" * 80)
    
    # 存储总统计结果
    total_train = 0
    total_val = 0
    
    # 存储所有文件名，用于检查重复
    train_filenames = set()
    val_filenames = set()
    
    # 遍历 train 目录
    train_dir = os.path.join(data_dir, "train")
    if os.path.exists(train_dir):
        print("=== train 目录 ===")
        for subdir in sorted(os.listdir(train_dir)):
            subdir_path = os.path.join(train_dir, subdir)
            if os.path.isdir(subdir_path):
                count, filenames = count_images(subdir_path)
                total_train += count
                train_filenames.update(filenames)
                print(f"  {subdir}: {count} 张图片")
        print(f"train 目录总计: {total_train} 张图片")
    else:
        print("train 目录不存在")
    
    print()
    
    # 遍历 val 目录
    val_dir = os.path.join(data_dir, "val")
    if os.path.exists(val_dir):
        print("=== val 目录 ===")
        for subdir in sorted(os.listdir(val_dir)):
            subdir_path = os.path.join(val_dir, subdir)
            if os.path.isdir(subdir_path):
                count, filenames = count_images(subdir_path)
                total_val += count
                val_filenames.update(filenames)
                print(f"  {subdir}: {count} 张图片")
        print(f"val 目录总计: {total_val} 张图片")
    else:
        print("val 目录不存在")
    
    print()
    
    # 检查重复图片
    print("=== 重复图片检查 ===")
    common_files = train_filenames & val_filenames
    if common_files:
        print(f"发现 {len(common_files)} 张重复图片:")
        for i, filename in enumerate(list(common_files)[:10]):  # 只显示前10个
            print(f"  {filename}")
        if len(common_files) > 10:
            print(f"  ... 还有 {len(common_files) - 10} 张重复图片")
    else:
        print("未发现重复图片")
    
    print()
    print("=== 总计 ===")
    print(f"train 目录: {total_train} 张图片")
    print(f"val 目录: {total_val} 张图片")
    print(f"总图片数: {total_train + total_val} 张图片")
    if common_files:
        print(f"重复图片数: {len(common_files)} 张图片")
        print(f"唯一图片数: {len(train_filenames | val_filenames)} 张图片")

if __name__ == "__main__":
    main()
