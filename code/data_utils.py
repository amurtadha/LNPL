
from torch.utils.data import Dataset
import  json
from tqdm import tqdm
import numpy as np


class process_pt(Dataset):
    def __init__(self,opt, fname, tokenizer,noisy=False):
        self.tokenizer=tokenizer
        self.max_seq_len=opt.max_seq_len
        labels = json.load(open('../../datasets/{0}/{1}/labels.json'.format( opt.task, opt.dataset)))
        data = json.load(open(fname,  encoding='utf-8', errors='ignore') )
        labels= {k:_ for _, k in enumerate(labels)}

        all_data=[]
        for d in tqdm(data):
            label =d['label']
            text =d['text']

            if opt.task =='STS':
                text2=d['text2']
                inputs = tokenizer.encode_plus(text.strip().lower(), d['text2'].strip().lower(), add_special_tokens=True,
                                               max_length=opt.max_seq_len, truncation='only_first',
                                               padding='max_length',
                                               return_token_type_ids=True)
            else:
                inputs = tokenizer.encode_plus(text.strip().lower(), None, add_special_tokens=True,
                                               max_length=opt.max_seq_len, truncation='only_first', padding='max_length',
                                               return_token_type_ids=True)

            input_ids = inputs['input_ids']
            input_mask = inputs['attention_mask']
            segment_ids = inputs["token_type_ids"]

            assert len(input_ids) <= opt.max_seq_len
            input_ids = np.asarray(input_ids, dtype='int64')
            input_mask = np.asarray(input_mask, dtype='int64')
            segment_ids = np.asarray(segment_ids, dtype='int64')
            if opt.task == 'STS':
                data = {
                    'text': text,
                    'text2': text2,
                    'input_ids': input_ids,
                    'segments_ids': segment_ids,
                    'input_mask': input_mask,
                    'label': labels[label],
                    'ori_label': labels[d['ori_label']]
                }
            else:
                data = {
                    'text': text,
                    'input_ids': input_ids,
                    'segments_ids': segment_ids,
                    'input_mask': input_mask,
                    'label': labels[label],
                    'ori_label': labels[d['ori_label']]
                }

            all_data.append(data)





        self.data = all_data


    def __getitem__(self, index):
        return self.data[index]
    # def update(self, index, value):
    #     self.data[index]['label'] =value

    def __len__(self):
        return len(self.data)
