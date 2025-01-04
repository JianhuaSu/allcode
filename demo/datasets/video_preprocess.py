from torchvision.io import read_image, read_video
from torchvision.models import resnet50, ResNet50_Weights, swin_b, Swin_B_Weights
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import *
import pickle
import os
import numpy as np


class VideoFeature:

    def __init__(self, args):

        print("**********视频特征提取开始**********")
        weights = Swin_B_Weights.IMAGENET1K_V1
        model = swin_b(weights=weights)

        train_nodes, eval_nodes = get_graph_node_names(model)

        return_nodes = {
            # node_name: user-specified key for output dict
            'flatten': 'layer1',
        }
        body = create_feature_extractor(model, return_nodes=return_nodes)
        preprocess = weights.transforms()

        s_list = os.listdir(args.raw_video_path)
        s_list = list(s_list)

        swin_feature = dict()
        for s in tqdm(s_list):
            vid, _, _ = read_video(os.path.join(args.raw_video_path, s), output_format="TCHW")
            video_name = s
            tmp = []
            for frame in tqdm(vid):       
                batch = preprocess(frame).unsqueeze(0)
                feature = body(batch)['layer1'].detach().numpy()
                tmp.append(feature)
            swin_feature[video_name] = np.expand_dims(np.squeeze(np.array(tmp), axis=1), axis=1)

        save_path = f'{args.video_data_path}/{args.video_feats}'

        with open(save_path, 'wb') as f:
            pickle.dump(swin_feature, f)
        print("**********视频特征提取结束**********")
