import torch
import torch.nn.functional as F
import logging
import os
from utils.functions import restore_model, get_intent_label, append_prediction_result
from backbones.FusionNets.MSZ import MMEncoder, MLP_head
__all__ = ['MSZ']

class MSZ:

    def __init__(self, args, data):
             
        self.logger = logging.getLogger(args.logger_name)
        
        self.device = args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.args = args
        
        self.tmm_data = data.data

        self.mm_encoder = MMEncoder(args).to(self.device)
        self.multiclass_classifier = MLP_head(args, args.num_labels).to(self.device)
        self.binary_classifier = MLP_head(args, 2).to(self.device)
        
        model_path_test = '/root/autodl-tmp/models/MSZ'
        self.mm_encoder = restore_model(self.mm_encoder, os.path.join(model_path_test, 'mm_encoder'), self.device)
        self.multiclass_classifier = restore_model(self.multiclass_classifier, os.path.join(model_path_test, 'multiclass_classifier'), self.device)
        self.binary_classifier = restore_model(self.binary_classifier, os.path.join(model_path_test, 'binary_classifier'), self.device)

    

    def _get_outputs(self, args, dataloader):
        
        self.binary_classifier.eval()
        self.mm_encoder.eval()
        self.multiclass_classifier.eval()

        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)

            
        with torch.set_grad_enabled(False):

            mix_fusion_feats, _ = self.mm_encoder(dataloader['text'],dataloader['video'], dataloader['audio'], ood_sampling = False)
            binary_logits = self.binary_classifier(mix_fusion_feats)
            binary_scores = F.softmax(binary_logits, dim = 1)[:, 1]

            logits = self.multiclass_classifier(mix_fusion_feats, binary_scores = binary_scores) 
                
            total_logits = torch.cat((total_logits, logits))

        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)

        y_pred = total_preds.cpu().numpy()

        return y_pred

    def _test(self, args):
        
        out = self._get_outputs(args, self.tmm_data)
        
        from utils.functions import get_intent_label, append_prediction_result
        
        bq = get_intent_label(out[0])
        
        append_prediction_result(args.method, bq)
        
        self.logger.info(f'\"{args.method}\" 方法的预测结果：\"{bq}\"')
