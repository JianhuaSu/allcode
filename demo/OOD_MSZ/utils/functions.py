
import os
import torch
import numpy as np
import random


def set_torch_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def restore_model(model, model_dir, device):
    output_model_file = os.path.join(model_dir, 'pytorch_model.bin')
    m = torch.load(output_model_file, map_location=device)
    model.load_state_dict(m, strict = False)
    return model

def get_intent_label(index):
    
    intent_labels = [
        'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize', 
        'Agree', 'Taunt', 'Flaunt', 
        'Joke', 'Oppose', 
        'Comfort', 'Care', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave', 
        'Prevent', 'Greet', 'Ask for help'
    ]
    
    # 确保输入的索引有效
    if 0 <= index < len(intent_labels):
        return intent_labels[index]
    else:
        return "Invalid index. Please enter a number between 0 and 19."

def append_prediction_result(method, bq):
        
    # 打开文件，使用追加模式 ('a')
    file_path = '/root/autodl-tmp/demo/datasets/结果.txt'
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f'\"{method}\"-\"{bq}\"\n')
        