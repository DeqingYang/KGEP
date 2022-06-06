# coding: utf-8

import os
import json
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from preprocess.douban_preprocess import read_from_pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


class TrainDataset(Dataset):
    def __init__(self, data_fn, node2id_fn, tt_type, tmp_fn='prep.pkl'):
        ''' 
        data_fn <str>: Filename of the douban dataset.
        node2id_fn <str>: Filename of the mapping (from kg indices to unified indices, e.g. {u0:0, u1:1, a0:2})
        '''
        super(TrainDataset, self).__init__()

            
        [data] = read_from_pickle(data_fn)

        with open(node2id_fn) as f:
            node2id = json.loads(f.read())
        
        # preprocess data
        data_u, data_i, lbls = [], [], []
        user_item_map = {}
        ind = 0
        for item in data:
            uid = node2id[item[0]]
            truei_id = node2id[item[1]]
            falsei_ids = [node2id[k] for k in item[3]]
            
            n_items = len(falsei_ids) + 1

            tmp_lbls = [0 for _ in range(n_items)]
            tmp_lbls[0] = 1

            data_u.extend([uid for _ in range(n_items)])
            data_i.append(truei_id)
            data_i.extend(falsei_ids)
            lbls.extend(tmp_lbls)
            user_item_map[uid] = [ind, ind+n_items]

            ind += n_items

        if tt_type == 'train':
            data_u = [data_u[i] for i in range(len(lbls)) if lbls[i]] * 9 + [data_u[i] for i in range(len(lbls)) if lbls[i] == 0]
            data_i = [data_i[i] for i in range(len(lbls)) if lbls[i]] * 9 + [data_i[i] for i in range(len(lbls)) if lbls[i] == 0]
            lbls = [lbls[i] for i in range(len(lbls)) if lbls[i]] * 9 + [lbls[i] for i in range(len(lbls)) if lbls[i] == 0]
        self.data = torch.stack([torch.tensor(data_u, device=device),
                             torch.tensor(data_i, device=device)])
        self.labels = torch.tensor(lbls, device=device)
        self.user_item_map = user_item_map
        
        with open(tmp_fn, 'wb') as f:
            pickle.dump([self.data, self.labels, self.user_item_map], f)

    def __len__(self):
        return self.data.size(1)

    def __getitem__(self, x):
        return self.data[:, x], self.labels[x]


class TestDataset(Dataset):
    def __init__(self, data_fn, node2id_fn, tt_type, tmp_fn='prep.pkl'):
        ''' 
        data_fn <str>: Filename of the douban dataset.
        node2id_fn <str>: Filename of the mapping (from kg indices to unified indices, e.g. {u0:0, u1:1, a0:2})
        '''
        super(TestDataset, self).__init__()
            
        [data] = read_from_pickle(data_fn)

        with open(node2id_fn) as f:
            node2id = json.loads(f.read())
        
        # preprocess data
        data_u, data_i, lbls = [], [], []
        user_item_map = {}
        ind = 0
        self.test_users=0
        for item in data:
            self.test_users+=1
            uid = node2id[item]
            truei_ids = [node2id[k] for k in data[item]['pos']]
            falsei_ids = [node2id[k] for k in data[item]['neg']]
            
            n_items = len(falsei_ids) + len(truei_ids)

            tmp_lbls = [0 for _ in range(n_items)]
            for j in range(len(truei_ids)):
                tmp_lbls[j] = 1

            data_u.extend([uid for _ in range(n_items)])
            data_i.extend(truei_ids)
            data_i.extend(falsei_ids)
            lbls.extend(tmp_lbls)
            user_item_map[uid] = [ind, ind+n_items]

            ind += n_items

        self.data = torch.stack([torch.tensor(data_u, device=device),
                             torch.tensor(data_i, device=device)])
        self.labels = torch.tensor(lbls, device=device)
        self.user_item_map = user_item_map
        
        with open(tmp_fn, 'wb') as f:
            pickle.dump([self.data, self.labels, self.user_item_map], f)

    def __len__(self):
        return self.data.size(1)

    def __getitem__(self, x):
        return self.data[:, x], self.labels[x]


def get_dataloader(dataset, batch_size=16, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


