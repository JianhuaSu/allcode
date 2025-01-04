import os
import csv
import torch
from transformers import BertTokenizer
import numpy as np
import pickle
import logging

class DataManager:
    
    def __init__(self, args):
        
        self.logger = logging.getLogger(args.logger_name)

        args.text_seq_len, args.video_seq_len, args.audio_seq_len  = 30, 230, 480
        args.text_feat_dim, args.video_feat_dim, args.audio_feat_dim = 768, 1024, 768

        args.ood_label_id = args.num_labels = 20
        self.data = prepare_data(args, args.data_path)


def get_t_data(args, data_path):

    processor = DatasetProcessor()

    test_examples = processor.get_examples(data_path)

    test_feats = get_backbone_feats(args, test_examples)

    return test_feats

def get_backbone_feats(args, examples):
    
    tokenizer = BertTokenizer.from_pretrained(args.text_pretrained_model, do_lower_case=True)   

    features = convert_examples_to_features(examples, 30, tokenizer)     
    features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in features]
    return torch.tensor(features_list).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text_a, text_b=None):

        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
                
            return lines

class DatasetProcessor(DataProcessor):

    def __init__(self):
        super(DatasetProcessor).__init__()
        
    def get_examples(self, data_dir):
        
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "text.tsv")))

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):

            text_a = line[0]
            examples.append(
                InputExample(text_a=text_a, text_b=None))
        return examples

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
        
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids)
                        )
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context
        else:
            tokens_b.pop()

          
def prepare_data(args, data_path):          

    ind_outputs = get_ind_data(args, data_path)     

    text_data, video_data, audio_data = ind_outputs['text_data'], ind_outputs['video_data'], ind_outputs['audio_data']
    
    data = {
        'text':text_data,
        'video':video_data,
        'audio':audio_data,
    }
    
    return data



def get_ind_data(args, data_path, ty='mm'):

    outputs = {}
    
    text_data = get_t_data(args, data_path)

    outputs['text_data'] = text_data

    if ty == 'text':
        return outputs
    
    else:
                        
        video_feats_path = os.path.join(data_path,'video_data', 'swin_feats.pkl')
        video_data = get_v_a_data(video_feats_path, 230)
        
        audio_feats_path = os.path.join(data_path,'audio_data', 'wavlm_feats.pkl')
        audio_data = get_v_a_data(audio_feats_path, 480)  
            
        outputs['video_data'] = video_data
        outputs['audio_data'] = audio_data
            
        return outputs


def get_v_a_data(feats_path, max_seq_len):
    
    if not os.path.exists(feats_path):
        raise Exception('Error: The directory of features is empty.')    

    feats = load_feats(feats_path)
    data = padding_feats(feats, max_seq_len)

    return torch.tensor(data['feats']).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
def load_feats(feats_path):

    with open(feats_path, 'rb') as f:
        feats = pickle.load(f)
        
    return feats

def padding(feat, max_length, padding_mode = 'zero', padding_loc = 'end'):
    """
    padding_mode: 'zero' or 'normal'
    padding_loc: 'start' or 'end'
    """
    assert padding_mode in ['zero', 'normal']
    assert padding_loc in ['start', 'end']

    length = feat.shape[0]
    
    if length > max_length:
        return feat[:max_length, :]

    if padding_mode == 'zero':
        pad = np.zeros([max_length - length, feat.shape[-1]])
    elif padding_mode == 'normal':
        mean, std = feat.mean(), feat.std()
        pad = np.random.normal(mean, std, (max_length - length, feat.shape[1]))
    
    if padding_loc == 'start':
        feat = np.concatenate((pad, feat), axis = 0)
    else:
        feat = np.concatenate((feat, pad), axis = 0)

    return feat

def padding_feats(feats, max_seq_len):
    
    p_feats = {}
    tmp_list = []
    length_list = []
    
    for dataset_type in feats.keys():
        f = feats[dataset_type]
        x_f = np.array(f) 
        x_f = x_f.squeeze(1) if x_f.ndim == 3 else x_f
        length_list.append(min(len(x_f), max_seq_len))
        p_feat = padding(x_f, max_seq_len)
        tmp_list.append(p_feat)

    p_feats = {
            'feats': tmp_list,
            'lengths': length_list
        }

    return p_feats    


