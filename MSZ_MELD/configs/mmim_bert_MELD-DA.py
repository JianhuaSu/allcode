class Param():
    
    def __init__(self, args):
        
        self.common_param = self._get_common_parameters(args)
        self.hyper_param = self._get_hyper_parameters(args)

    def _get_common_parameters(self, args):
        """
            padding_mode (str): The mode for sequence padding ('zero' or 'normal').
            padding_loc (str): The location for sequence padding ('start' or 'end'). 
            eval_monitor (str): The monitor for evaluation ('loss' or metrics, e.g., 'f1', 'acc', 'precision', 'recall').  
            need_aligned: (bool): Whether to perform data alignment between different modalities.
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patience (int): Patient steps for Early Stop.
        """
        common_parameters = {
            'padding_mode': 'zero',
            'padding_loc': 'end',
            'need_aligned': False,
            'eval_monitor': 'f1',
            'train_batch_size': 16,
            'eval_batch_size': 8,
            'test_batch_size': 8,
            'wait_patience': 8,
        }
        return common_parameters

    def _get_hyper_parameters(self, args):

        hyper_parameters = {
            'num_train_epochs': 100,
            'add_va': False,
            'cpc_activation': 'Tanh',
            'mmilb_mid_activation': 'ReLU',
            'mmilb_last_activation': 'Tanh',
            'optim': 'Adam',
            'contrast': True,
            'bidirectional': True,
            'grad_clip': 1.0,
            'lr_main': [0.0001],
            'weight_decay_main': [0.0001],
            'lr_bert': [4e-5],
            'weight_decay_bert': [7e-5],
            'lr_mmilb': 0.001,
            'weight_decay_mmilb': 0.0001,
            'alpha': 0.1,
            'dropout_a': 0.1,
            'dropout_v': 0.1,
            'dropout_prj': 0.1,
            'n_layer': 1,
            'cpc_layers': 1,
            'd_vh': [32],
            'd_ah': [32],
            'd_vout': [16],
            'd_aout': [16],
            'd_prjh': [128],
            'scale': 20,
            'beta':0.5
        }
        return hyper_parameters