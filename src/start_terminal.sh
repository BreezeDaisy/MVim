#!/bin/sh

# 加载conda环境。每次切换环境时，都需要先加载conda环境
. /home/zdx/anaconda3/etc/profile.d/conda.sh

# 再激活所需的环境，出现环境
conda activate EMP

# 启动sh终端，启动子进程继承之前的环境
exec sh