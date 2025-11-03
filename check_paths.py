import sys
import os

print("当前工作目录:", os.getcwd())
print("\nPython导入路径:")
for i, path in enumerate(sys.path):
    print(f"[{i}] {path}")