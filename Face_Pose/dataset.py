import os
import shutil
import random
import gc

# 源目录和目标目录
SOURCE_DIR = '/home/zdx/python_daima/MVim/MVim/Face_Pose/Constructed_Small_sample_0.85'
TARGET_DIR = '/home/zdx/python_daima/MVim/MVim/Face_Pose/small_data'

# 创建目标目录结构
def create_directory_structure():
    os.makedirs(os.path.join(TARGET_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, 'val'), exist_ok=True)

# 合并 train 和 val 目录
def merge_train_val():
    # 遍历所有子目录
    for subdir in os.listdir(os.path.join(SOURCE_DIR, 'train')):
        subdir_path = os.path.join(SOURCE_DIR, 'train', subdir)
        if os.path.isdir(subdir_path):
            # 创建目标子目录
            target_train_subdir = os.path.join(TARGET_DIR, 'train', subdir)
            os.makedirs(target_train_subdir, exist_ok=True)
            
            # 复制 train 中的文件
            for file in os.listdir(subdir_path):
                src_file = os.path.join(subdir_path, file)
                dst_file = os.path.join(target_train_subdir, file)
                shutil.copy2(src_file, dst_file)
            
            # 复制 val 中的文件（如果存在）
            val_subdir_path = os.path.join(SOURCE_DIR, 'val', subdir)
            if os.path.exists(val_subdir_path):
                for file in os.listdir(val_subdir_path):
                    src_file = os.path.join(val_subdir_path, file)
                    dst_file = os.path.join(target_train_subdir, file)
                    shutil.copy2(src_file, dst_file)

# 从合并后的 train 中抽取15%移动到 val，再复制6%到 val
def split_train_val():
    for subdir in os.listdir(os.path.join(TARGET_DIR, 'train')):
        subdir_path = os.path.join(TARGET_DIR, 'train', subdir)
        if os.path.isdir(subdir_path):
            # 创建 val 子目录
            target_val_subdir = os.path.join(TARGET_DIR, 'val', subdir)
            os.makedirs(target_val_subdir, exist_ok=True)
            
            # 获取所有文件
            files = os.listdir(subdir_path)
            random.shuffle(files)
            
            # 计算要移动和复制的文件数量
            total_files = len(files)
            move_count = int(total_files * 0.15)
            copy_count = int(total_files * 0.06)
            
            # 移动 15% 的文件到 val 目录
            for file in files[:move_count]:
                src_file = os.path.join(subdir_path, file)
                dst_file = os.path.join(target_val_subdir, file)
                shutil.move(src_file, dst_file)
            
            # 复制 6% 的文件到 val 目录
            for file in files[move_count:move_count + copy_count]:
                src_file = os.path.join(subdir_path, file)
                dst_file = os.path.join(target_val_subdir, file)
                shutil.copy2(src_file, dst_file)

# 主函数
def main():
    print("开始处理数据集...")
    
    # 创建目录结构
    create_directory_structure()
    print("创建目录结构完成")
    
    # 合并 train 和 val
    merge_train_val()
    print("合并 train 和 val 完成")
    
    # 分割 train 和 val
    split_train_val()
    print("分割 train 和 val 完成")
    
    # 释放缓存
    gc.collect()
    print("释放缓存完成")
    
    print("数据集处理完成！")

if __name__ == "__main__":
    main()
