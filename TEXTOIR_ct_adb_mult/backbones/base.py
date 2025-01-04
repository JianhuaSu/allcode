import torch
import logging
from torch import nn
from .FusionNets import multimodal_methods_map

__all__ = ['ModelManager']

class MIA(nn.Module):

    def __init__(self, args, t=None, a = None, v = None):

        super(MIA, self).__init__()

        fusion_method = multimodal_methods_map[args.multimodal_method]
        self.model = fusion_method(args)
        
    def forward(self, text_feats, video_data, audio_data, *args, **kwargs):
        
        mm_model = self.model(text_feats, video_data, audio_data, *args, **kwargs)

        return mm_model


        
class ModelManager:

    def __init__(self, args):
        
        self.logger = logging.getLogger(args.logger_name)
        self.device = args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.model = self._set_model(args)

    def _set_model(self, args):

        model = MIA(args) 
        model.to(self.device)

        return model