import torch
import torch.nn.functional as F
import logging
from torch import nn
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from data.utils import get_dataloader
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from losses import loss_map
from .boundary import BoundaryLoss
from .pretrain import PretrainManager
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import os
from utils.metrics import AverageMeter
from utils.metrics import AverageMeter, Metrics, OOD_Metrics

__all__ = ['MULT']


class MULT:

    def __init__(self, args, data, model):

        self.logger = logging.getLogger(args.logger_name)
        
        self.device = model.device

        pretrain_model = PretrainManager(args, data, model)
        self.model = pretrain_model.model
        
        mm_data = data.data
        mm_dataloader = get_dataloader(args, mm_data)
        
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            mm_dataloader['train'], mm_dataloader['dev'], mm_dataloader['test']
        self.centroids = None
        self.metrics = Metrics(args)

        self.args = args
        self.criterion = nn.CrossEntropyLoss()

        if args.train:
            self.delta = None
            self.delta_points = []
        else:
            self.best_eval_score = 0
            self.delta = np.load(os.path.join(args.model_output_path, 'deltas.npy'))
            self.delta = torch.from_numpy(self.delta).to(self.device)
            self.centroids = np.load(os.path.join(args.model_output_path, 'centroids.npy'))
            self.centroids = torch.from_numpy(self.centroids).to(self.device)
            self.model = restore_model(self.model, args.model_output_path, self.device)
            

    def class_count(self, labels):
        class_data_num = []
        for l in np.unique(labels):
            num = len(labels[labels == l])
            class_data_num.append(num)
        return class_data_num
    
    
    def centroids_cal(self, model, args, train_dataloader, device):
        
        model.eval()
        centroids = torch.zeros(args.num_labels, 768).to(device)
        total_labels = torch.empty(0, dtype=torch.long).to(device)

        with torch.set_grad_enabled(False):

            for batch in tqdm(self.train_dataloader, desc="Iteration"):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)
                
                pooled_output, n_logits, c_logits = self.model(text_feats, video_feats, audio_feats)
                
                total_labels = torch.cat((total_labels, label_ids))

                for i in range(len(label_ids)):
                    label = label_ids[i]
                    centroids[label] += pooled_output[i]
                
        total_labels = total_labels.cpu().numpy()
        centroids /= torch.tensor(self.class_count(total_labels)).float().unsqueeze(1).to(device)
        
        return centroids

    def open_classify(self, args, features):
        
        logits = self.euclidean_metric(features, self.centroids)
        probs, preds = F.softmax(logits.detach(), dim = 1).max(dim = 1)
        euc_dis = torch.norm(features - self.centroids[preds], 2, 1).view(-1)
        preds[euc_dis >= self.delta[preds]] = args.num_labels
        
        return preds
    
    def euclidean_metric(self, a, b):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        logits = -((a - b)**2).sum(dim=2)
        return logits


    def F_measure(self, cm):
        idx = 0
        rs, ps, fs = [], [], []
        n_class = cm.shape[0]
        
        for idx in range(n_class):
            TP = cm[idx][idx]
            r = TP / cm[idx].sum() if cm[idx].sum() != 0 else 0
            p = TP / cm[:, idx].sum() if cm[:, idx].sum() != 0 else 0
            f = 2 * r * p / (r + p) if (r + p) != 0 else 0
            rs.append(r * 100)
            ps.append(p * 100)
            fs.append(f * 100)
            
        f = np.mean(fs).round(4)
        f_seen = np.mean(fs[:-1]).round(4)
        f_unseen = round(fs[-1], 4)
        
        results = {}
        results['F1-known'] = f_seen
        results['F1-open'] = f_unseen
        results['F1'] = f
        
        return results
    
    
    def _train(self, args): 

        early_stopping = EarlyStopping(args)
        
        criterion_boundary = BoundaryLoss(args, num_labels = args.num_labels, feat_dim = 768, neg=True, device = self.device)
        
        self.delta = F.softplus(criterion_boundary.delta)
        self.delta_points.append(self.delta)
        
        optimizer = torch.optim.Adam(criterion_boundary.parameters(), lr = 0.05)
        
        if self.centroids is None:
            self.centroids = self.centroids_cal(self.model, args, self.train_dataloader, self.device)

        # Collect features for known classes
        known_features = []
        known_labels = []
        self.model.eval()
        with torch.no_grad():
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)
                pooled_output, n_logits, c_logits = self.model(text_feats, video_feats, audio_feats)

                known_features.append(pooled_output.cpu().numpy())
                known_labels.append(label_ids.cpu().numpy())
        
        known_features = np.concatenate(known_features, axis=0)
        known_labels = np.concatenate(known_labels, axis=0)
        

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            loss_record = AverageMeter()
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                with torch.no_grad():                    
                    pooled_output, n_logits, c_logits = self.model(text_feats, video_feats, audio_feats)
                
                known_features_tensor = torch.tensor(known_features, device=self.device)
                known_labels_tensor = torch.tensor(known_labels, device=self.device)
                neg_features = []

                for i, (feature, label) in enumerate(zip(pooled_output, label_ids.cpu().numpy())):
                    # Convert feature to tensor
                    feature_tensor = torch.tensor(feature, device=self.device)
                    
                    # Get features from all other known classes
                    other_known_mask = known_labels_tensor != label
                    other_known_features = known_features_tensor[other_known_mask]
                    
                    # Calculate distances to all other samples
                    distances = torch.norm(other_known_features - feature_tensor, dim=1)
                    nearest_neg_idx = torch.argmin(distances)
                    neg_features.append(other_known_features[nearest_neg_idx])
                
                neg_features = torch.stack(neg_features)
                
                # Combine positive and negative examples, maintaining the pairing relationship
                combined_features = torch.stack([torch.tensor(pooled_output).to(self.device), neg_features], dim=1)     
                                    
                    
                with torch.set_grad_enabled(True):

                    loss, self.delta = criterion_boundary(combined_features, self.centroids, label_ids)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            outputs = self._get_outputs(args, mode = 'eval')
            eval_score = round(f1_score(outputs['y_true'], outputs['y_pred'], average='macro') * 100, 2)

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
            
            np.save(os.path.join(args.model_output_path, 'centroids.npy'), self.centroids.detach().cpu().numpy())
            np.save(os.path.join(args.model_output_path, 'deltas.npy'), self.delta.detach().cpu().numpy())


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
                
                pooled_output, n_logits, c_logits = self.model(text_feats, video_feats, audio_feats)
                
                preds = self.open_classify(args, pooled_output)
                total_preds = torch.cat((total_preds, preds))
                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, pooled_output))
                
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        
        if test_ind:
            outputs = self.metrics(y_true[y_true != args.ood_label_id], y_pred[y_true != args.ood_label_id])
        else:
            outputs = self.metrics(y_true, y_pred, show_results = show_results)
        
        outputs.update(
            {
                'y_true': y_true,
                'y_pred': y_pred,
            }
        )
        
        return outputs

    def _test(self, args, show=True):
        
        test_results = {}

        outputs = self._get_outputs(args, mode = 'test')
        
        test_results.update(outputs)

        self.logger.info("***** Test results *****")
            
        for key in sorted(test_results.keys()):
            self.logger.info("  %s = %s", key, str(test_results[key]))

        return test_results
    