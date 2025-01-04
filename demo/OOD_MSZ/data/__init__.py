

benchmarks = {
    'MIntRec':{
        'intent_labels': [
                    'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize', 
                    'Agree', 'Taunt', 'Flaunt', 
                    'Joke', 'Oppose', 
                    'Comfort', 'Care', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave', 
                    'Prevent', 'Greet', 'Ask for help' 
        ],
        'binary_maps': {
                    'Complain': 'Emotion', 'Praise':'Emotion', 'Apologise': 'Emotion', 'Thank':'Emotion', 'Criticize': 'Emotion',
                    'Care': 'Emotion', 'Agree': 'Emotion', 'Taunt': 'Emotion', 'Flaunt': 'Emotion',
                    'Joke':'Emotion', 'Oppose': 'Emotion', 
                    'Inform':'Goal', 'Advise':'Goal', 'Arrange': 'Goal', 'Introduce': 'Goal', 'Leave':'Goal',
                    'Prevent':'Goal', 'Greet': 'Goal', 'Ask for help': 'Goal', 'Comfort': 'Goal'
        },
        'binary_intent_labels': ['Emotion', 'Goal'],

        'max_seq_lengths': {
            'text': 30,
            'video': 230, 
            'audio': 480
        },
        'feat_dims': {
            'text': 768,
            'video': 1024,
            'audio': 768
        },
        'ood_data':{
            'MIntRec-OOD': {'ood_label': 'UNK'},
        }
    },
    
    'MIntRec2': {
        'intent_labels': [
            'Acknowledge', 'Advise', 'Agree', 'Apologise', 'Arrange', 
            'Ask for help', 'Asking for opinions', 'Care', 'Comfort', 'Complain', 
            'Confirm', 'Criticize', 'Doubt', 'Emphasize', 'Explain', 
            'Flaunt', 'Greet', 'Inform', 'Introduce', 'Invite', 
            'Joke', 'Leave', 'Oppose', 'Plan', 'Praise', 
            'Prevent', 'Refuse', 'Taunt', 'Thank', 'Warn',
        ],
        'max_seq_lengths': {
            'text': 50,
            'video': 180,
            'audio': 400
        },
        'feat_dims': {
            'text': 768,
            'video': 256,
            'audio': 768
        },
        'ood_data':{
            'MIntRec2-OOD':{'ood_label': 'UNK'}
        }
    },
    'MELD-DA':{
        'intent_labels':['a', 'ag', 'ans', 'ap', 'b', 'c', 'dag', 'g', 'o', 'q', 's'],
        'max_seq_lengths': {
            'text': 70, # max: 69, final: 27
            'video': 250, ### max: 618, final: 242
            'audio': 530 ### max 2052, final: 524
        },
        'feat_dims': {
            'text': 768,
            'video': 1024,
            'audio': 768
        },
        'ood_data':{
            'MELD-DA-OOD': {'ood_label': 'oth'},
        }
    },
}

