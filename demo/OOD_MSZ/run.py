from configs.base import ParamManager, add_config_param
from data.base import DataManager
from methods import method_map
from backbones.base import ModelManager
from utils.functions import set_torch_seed

import argparse
import logging
import os
import datetime
import warnings

def parse_arguments():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logger_name', type=str, default='Multimodal Intent Recognition', help="Logger name for multimodal intent recognition.")
        
    parser.add_argument('--method', type=str, default='MSZ', help="which method to use.")

    parser.add_argument("--text_backbone", type=str, default='bert-base-uncased', help="which backbone to use for text modality")

    parser.add_argument("--text_backbone_path", type=str, default='/root/autodl-tmp/demo/mode/bert-base-uncased', help="which backbone to use for text modality")

    parser.add_argument('--seed', type=int, default=0, help="The selected person id.")

    parser.add_argument('--num_workers', type=int, default=8, help="The number of workers to load data.")

    parser.add_argument('--log_id', type=str, default=None, help="The index of each logging file.")
    
    parser.add_argument('--gpu_id', type=str, default='0', help="The selected person id.")

    parser.add_argument("--data_path", default = '/demo/datasets', type=str,
                        help="The input data dir. Should contain text, video and audio data for the task.")

    parser.add_argument('--log_path', type=str, default='logs', help="Logger directory.")

    args = parser.parse_args()

    return args

def set_logger(args):
    
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.logger_name =  f"{args.method}"
    args.log_id = f"{args.logger_name}_{time}"
    
    logger = logging.getLogger(args.logger_name)
    logger.setLevel(logging.DEBUG)

    log_path = os.path.join(args.log_path, args.log_id + '.log')
    fh = logging.FileHandler(log_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    
    return logger

    
def work(args, data, logger, debug_args=None):
    
    set_torch_seed(args.seed)
    
    method_manager = method_map[args.method]
    
    if args.method.startswith(('msz')):
        method = method_manager(args, data)
    else:
        model = ModelManager(args)
        method = method_manager(args, data, model)
        
    outputs = method._test(args)    


if __name__ == '__main__':
    
    warnings.filterwarnings('ignore')
    
    args = parse_arguments()
    logger = set_logger(args)
     
    param = ParamManager(args)
    args = param.args
    
    data = DataManager(args)
        
    args.seed = 0
    args = add_config_param(args, args.method)

    work(args, data, logger)

        

