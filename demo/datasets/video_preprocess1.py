from torchvision.io import read_image, read_video
from torchvision.models import resnet50, ResNet50_Weights, swin_b, Swin_B_Weights
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import *
import pickle
import os
import numpy as np

# with open('speaker_annotation.pkl', 'rb') as f:
#     speaker_annotation = pickle.load(f)

weights = Swin_B_Weights.IMAGENET1K_V1
model = swin_b(weights=weights)

train_nodes, eval_nodes = get_graph_node_names(model)

return_nodes = {
    # node_name: user-specified key for output dict
    'flatten': 'layer1',
}
body = create_feature_extractor(model, return_nodes=return_nodes)
preprocess = weights.transforms()

data_path = '/root/autodl-tmp/demo/datasets/video_data/raw_data'

s_list = os.listdir(data_path)
s_list = list(s_list)
# half_len = int(len(s_list) / 2)
# s_list = s_list[:half_len]


swin_feature = dict()
for s in tqdm(s_list):
    vid, _, _ = read_video(os.path.join(data_path, s), output_format="TCHW")
    print(vid.shape)
    video_name = s
    tmp = []
    for frame in tqdm(vid):       
        batch = preprocess(frame).unsqueeze(0)
        feature = body(batch)['layer1'].detach().numpy()
        tmp.append(feature)
    swin_feature[video_name] = np.expand_dims(np.squeeze(np.array(tmp), axis=1), axis=1)

save_path = '/root/autodl-tmp/demo/datasets/video_data/swin_feats.pkl'

with open(save_path, 'wb') as f:
    pickle.dump(swin_feature, f)
