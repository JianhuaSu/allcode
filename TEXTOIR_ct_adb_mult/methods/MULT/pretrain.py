import torch
import torch.nn.functional as F
import logging
from torch import nn
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from utils.metrics import AverageMeter, Metrics, OOD_Metrics, OID_Metrics
from data.utils import get_dataloader
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from losses import loss_map



class PretrainManager:

    def __init__(self, args, data, model):

        self.logger = logging.getLogger(args.logger_name)
        
        self.device = model.device
        self.model = model._set_model(args)

        self.optimizer = optim.Adam(self.model.parameters(), lr = args.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, verbose=True, patience=args.wait_patience)
        self.triplet_criterion = nn.TripletMarginLoss(margin=args.margin, p=args.p)
        self.contrast_criterion = loss_map['SupConLoss'] 
        mm_data = data.data
        mm_dataloader = get_dataloader(args, mm_data)
        
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            mm_dataloader['train'], mm_dataloader['dev'], mm_dataloader['test']
        
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = Metrics(args)
        self.oid_metrics = OID_Metrics(args)
        self.ood_metrics = OOD_Metrics(args)

        if args.pretrain:
            self.logger.info('Pre-training Begin...')
            self.best_eval_score = 0
            self._train(args)
        else:
            self.model = restore_model(self.model, args.model_output_path, self.device)

    def _train(self, args): 

        early_stopping = EarlyStopping(args)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            loss_record = AverageMeter()
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                with torch.set_grad_enabled(True):

                    pooled_output, n_logits, c_logits = self.model(text_feats, video_feats, audio_feats)
                    
                    n_loss = self.criterion(n_logits, label_ids)

                    batch_size = pooled_output.shape[0]
                    
                    # 原有的SupConLoss计算
                    norm_logits = F.normalize(n_logits)
                    norm_logits_aug = F.normalize(c_logits)
                    t_label_ids = label_ids.expand(batch_size, batch_size)
                    mask = torch.eq(t_label_ids, t_label_ids.T).long()
                    logits_mask = torch.scatter(
                        mask,
                        0,
                        torch.arange(batch_size).unsqueeze(0).to(self.device),
                        1
                    )
                    contrastive_logits = torch.cat((norm_logits.unsqueeze(1), norm_logits_aug.unsqueeze(1)), dim=1)
                    supcon_loss = self.contrast_criterion(contrastive_logits, mask=logits_mask, temperature=0.3, device=self.device)
                    
                    # 新增的三元组损失计算
                    dist_matrix = torch.cdist(pooled_output, pooled_output)
                    anchors, positives, negatives = [], [], []
                    for i in range(batch_size):
                        anchor = pooled_output[i]
                        anchor_label = label_ids[i]
                        
                        # 选择同类别中最远的样本作为正样本
                        pos_candidates = (label_ids == anchor_label) & (torch.arange(batch_size).to(self.device) != i)
                        if pos_candidates.sum() > 0:
                            pos_index = dist_matrix[i][pos_candidates].argmax()
                            positive = pooled_output[pos_candidates][pos_index]
                        else:
                            positive = anchor  # 如果没有其他同类样本，使用自身
                        
                        # 选择不同类别中最近的样本作为负样本
                        neg_candidates = label_ids != anchor_label
                        if neg_candidates.sum() > 0:
                            neg_index = dist_matrix[i][neg_candidates].argmin()
                            negative = pooled_output[neg_candidates][neg_index]
                        else:
                            negative = pooled_output[(torch.arange(batch_size) != i).nonzero().squeeze()][0]  # 随机选择一个不同的样本
                        
                        anchors.append(anchor)
                        positives.append(positive)
                        negatives.append(negative)
                    
                    anchors = torch.stack(anchors)
                    positives = torch.stack(positives)
                    negatives = torch.stack(negatives)
                    
                    triplet_loss = self.triplet_criterion(anchors, positives, negatives)
                    
                    # 结合所有损失
                    loss = args.loss_weight * n_loss + \
                           args.supcon_loss_weight * supcon_loss + \
                           args.triplet_loss_weight * triplet_loss
                           
                           
                    self.optimizer.zero_grad()
                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))
                    self.optimizer.step()
            
            outputs = self._get_outputs(args, mode = 'eval')
            self.scheduler.step(outputs['loss'])
            eval_score = outputs[args.eval_monitor]

            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
                'eval_score': round(eval_score, 4),
            }

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            early_stopping(eval_score, self.model)

            if early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

        self.best_eval_score = early_stopping.best_score
        self.model = early_stopping.best_model   
        
        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)   

    def _get_outputs(self, args, mode = 'eval', show_results = False ,test_ind = False):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        total_features = torch.empty((0, 768)).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        
        loss_record = AverageMeter()

        for batch in tqdm(dataloader, desc="Iteration"):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            
            with torch.set_grad_enabled(False):
                
                pooled_output, logits, c_logits = self.model(text_feats, video_feats, audio_feats)

                total_logits = torch.cat((total_logits, logits))
                total_features = torch.cat((total_features, pooled_output))
                total_labels = torch.cat((total_labels, label_ids))

                if mode == 'eval':
                    loss = self.criterion(logits, label_ids)
                    loss_record.update(loss.item(), label_ids.size(0))

        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)
        
        y_logit = total_logits.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        y_feat = total_features.cpu().numpy()
        y_prob = total_maxprobs.cpu().numpy()

        outputs = self.metrics(y_true, y_pred, show_results=show_results)

        if test_ind:
            outputs = self.metrics(y_true[y_true != args.ood_label_id], y_pred[y_true != args.ood_label_id])
        else:
            outputs = self.metrics(y_true, y_pred, show_results = show_results)

        if mode == 'eval':
            outputs.update({'loss': loss_record.avg})

        outputs.update(
            {
                'y_feat': y_feat,
                'y_logit': y_logit,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_prob': y_prob,
            }
        )

        return outputs

    