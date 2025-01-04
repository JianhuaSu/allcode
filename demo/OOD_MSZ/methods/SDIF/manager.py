import torch
import torch.nn.functional as F
import logging
from utils.functions import restore_model

class SDIF:

    def __init__(self, args, data, model):
             
        self.logger = logging.getLogger(args.logger_name)
        
        self.logger = logging.getLogger(args.logger_name)
        
        self.device, self.model = model.device, model.model

        self.tmm_data = data.data
        
        self.args = args
  
        model_path_test = '/root/autodl-tmp/models/SDIF'
        self.model = restore_model(self.model, model_path_test, self.device)
            
            
    def _get_outputs(self, args, dataloader):

        self.model.eval()

        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)

        with torch.set_grad_enabled(False):
                
            logits = self.model(dataloader['text'],dataloader['video'], dataloader['audio'])
                
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
        
        self.logger.info(f'{args.method} 方法的预测结果：{out[0]}')

