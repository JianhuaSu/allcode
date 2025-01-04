import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from ..SubNets.transformers_encoder.transformer import TransformerEncoder
from torch import nn
from ..SubNets import text_backbones_map

__all__ = ['MULT']

class MULT(nn.Module):
    
    def __init__(self, args):

        super(MULT, self).__init__()

        text_backbone = text_backbones_map[args.text_backbone]

        self.text_subnet = text_backbone(args)
        self.linear_layer = nn.Linear(768*2+256, 768)  # 定义线性层

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        self.n_classifier = nn.Linear(768, args.num_labels)
        self.c_classifier = nn.Linear(768, args.num_labels)

    def forward(self, text_feats, video_feats, audio_feats):
        
        text = self.text_subnet(text_feats)
        
        audio_feats, video_feats = audio_feats.float(), video_feats.float()

        text = torch.mean(text, dim=1)  # 根据第 1 维度取平均，输出 [32, 768]
        video_feats = torch.mean(video_feats, dim=1)  # 根据第 1 维度取平均，输出 [32, 1024]
        audio_feats = torch.mean(audio_feats, dim=1)  # 根据第 1 维度取平均，输出 [32, 768]
        
        combined_feats = torch.cat((text, video_feats, audio_feats), dim=1)  # 拼接，输出 [32, 2560]
        final_feats = self.linear_layer(combined_feats)  # 输出 [32, 768]
        final_feats = self.dropout(self.activation(final_feats))

        n_logits = self.n_classifier(final_feats)
        c_logits = self.c_classifier(final_feats)
        
        return final_feats, n_logits, c_logits


    
    