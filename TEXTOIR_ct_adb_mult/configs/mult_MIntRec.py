class Param():
    
    def __init__(self, args):
        
        self.hyper_param = self._get_hyper_parameters(args)

    def _get_hyper_parameters(self, args):
        """
        Args:
            num_train_epochs (int): The number of training epochs.
            dst_feature_dims (int): The destination dimensions (assume d(l) = d(v) = d(t)).
            nheads (int): The number of heads for the transformer network.
            n_levels (int): The number of layers in the network.
            attn_dropout (float): The attention dropout.
            attn_dropout_v (float): The attention dropout for the video modality.
            attn_dropout_a (float): The attention dropout for the audio modality.
            relu_dropout (float): The relu dropout.
            embed_dropout (float): The embedding dropout.
            res_dropout (float): The residual block dropout.
            output_dropout (float): The output layer dropout.
            text_dropout (float): The dropout for text features.
            grad_clip (float): The gradient clip value.
            attn_mask (bool): Whether to use attention mask for Transformer. 
            conv1d_kernel_size_l (int): The kernel size for temporal convolutional layers (text modality).  
            conv1d_kernel_size_v (int):  The kernel size for temporal convolutional layers (video modality).  
            conv1d_kernel_size_a (int):  The kernel size for temporal convolutional layers (audio modality).  
            lr (float): The learning rate of backbone.
        """

        hyper_parameters = {
            'padding_mode': 'zero',
            'padding_loc': 'end',
            'need_aligned': False,
            'eval_monitor': 'acc',
            'train_batch_size': 32,
            'eval_batch_size': 16,
            'test_batch_size': 16,
            'wait_patience': 5,
            'num_train_epochs': 1,

            'lr': [0.00003],  # 5e-
            
            'safe1': [0.7], 
            'safe2': [0.7], 
            'neg_weight': [0.5],
            
            'margin': [0.6],
            'p': 2,
            'loss_weight': 1,
            'supcon_loss_weight': [0.6],
            'triplet_loss_weight': [0.6]
        }
        return hyper_parameters

