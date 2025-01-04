#!/bin/bash

# 定义文件路径
RESULT_FILE="/root/autodl-tmp/demo/datasets/结果.txt"

# 检查文件是否存在
if [ -f "$RESULT_FILE" ]; then
    rm -f "$RESULT_FILE"  # 删除文件
fi

# 创建新的文件
touch "$RESULT_FILE"


for method in 'msz' 'sdif' 'misa' 'mag_bert' 'mult' 
do
    CUDA_VISIBLE_DEVICES=0 python run.py \
    --data_path '/root/autodl-tmp/demo/datasets' \
    --logger_name ${method} \
    --method ${method}\
    --gpu_id '0' 
done

