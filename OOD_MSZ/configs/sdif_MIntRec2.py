class Param():
    
    def __init__(self, args):

        self.hyper_param = self._get_hyper_parameters(args)

    def _get_hyper_parameters(self, args):
        '''
        Args:
            num_train_epochs (int): The number of training epochs.
            num_labels (autofill): The output dimension.
            max_seq_length (autofill): The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
            freeze_backbone_parameters (binary): Whether to freeze all parameters but the last layer.
            feat_dim (int): The feature dimension.
            warmup_proportion (float): The warmup ratio for learning rate.
            activation (str): The activation function of the hidden layer (support 'relu' and 'tanh').
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patient (int): Patient steps for Early Stop.
        '''
        if args.text_backbone.startswith('bert'):

            hyper_parameters = {
                'data_mode': 'multi-class',
                'padding_mode': 'zero',
                'padding_loc': 'end',
                'need_aligned': False,
                'eval_monitor': ['f1'],
                'train_batch_size': [16],
                'eval_batch_size': 8,
                'test_batch_size': 8,
                'wait_patience': [8],
                'num_train_epochs': [100],
                'dst_feature_dims': 768,
                
                'n_levels_self': 1,
                'n_levels_cross': 1,
                'dropout_rate': 0.2,
                'cross_dp_rate': 0.3,
                'cross_num_heads': 12,
                'self_num_heads': 8,
                'grad_clip': 7, 
                'lr': [9e-6], #9e-6原始
                'opt_patience': 8,
                'factor': 0.5,
                'weight_decay': 0.01,
                'aug_lr': 1e-6, #1e-6原始
                'aug_epoch': 1,
                'aug_dp': 0.3,
                'aug_weight_decay': 0.1,
                'aug_grad_clip': -1.0,
                'aug_batch_size': 16,
                'aug': True,
                'aug_num': 25000,
                'use_wandb': False,
                'scale': [16],
            }
           

        return hyper_parameters