# coding: utf-8

import os
import json
import torch
import numpy as np
from ipdb import set_trace
import random

class environment:
    def __init__(self, graph_map, self_loop_idx, node2id_fn='./data/book/node2id.json', resample_to_n=200):

        if os.path.exists(node2id_fn):
            with open(node2id_fn) as f:
                self.target_dict = json.loads(f.read())

        self.resample_to_n = resample_to_n
        self.self_loop_idx = self_loop_idx

        # indexify maps
        new_map = {}
        for k in ['u', 'i', 'a']:
            for key in graph_map[k]:
                new_key = self.target_dict[key]
                new_map[new_key] = [[i[0], self.target_dict[i[1]]] for i in graph_map[k][key]]
                while len(new_map[new_key]) < self.resample_to_n:
                    new_map[new_key] += [[i[0], self.target_dict[i[1]]] for i in graph_map[k][key]]
                    
        for k in ['u', 'i', 'a']:
            for key in graph_map[k]:
#                 new_key = self.target_dict[key]
                 for i in graph_map[k][key]:
                        if not new_map.get(self.target_dict[i[1]]):
                            new_map[self.target_dict[i[1]]]=[[self.self_loop_idx,self.target_dict[i[1]]]]*self.resample_to_n 
                          

        self.new_map = new_map
        self.create_new_map_emb()

    def create_new_map_emb(self):
        self.new_map_emb = torch.tensor([random.sample(self.new_map[k], self.resample_to_n)
                                         if k in self.new_map else
                                         random.sample(self.new_map.get(k, []) + self.new_map[0], self.resample_to_n)
                                         for k in range(max(self.new_map.keys()) + 1)], dtype=torch.long).cuda()
    

    def get_action_new(self, indices):
        batch_action = self.new_map_emb[indices]
        return batch_action[:, :, 1], batch_action[:, :, 0]


    def get_action(self, indices, resample_to_n=200):
        ''' Batched version.
        '''
        B = indices.size(0)
        node_inds_list = []
        edge_types_list = []

        for i in range(B):
            n, e = self._get_action(indices[i], resample_to_n)
            node_inds_list.append(n)
            edge_types_list.append(e)
        return torch.stack(node_inds_list), torch.stack(edge_types_list)

    def _get_action(self, index, resample_to_n=200):
        ''' Find next possible actions. Use resampling to reorder number of candidates.
        '''
        # indices [B, 1]
        index = int(index)
        next_data = self.new_map[index]
        next_data = torch.tensor(next_data, dtype=torch.long)
        node_inds, edge_types = next_data[:, 1], next_data[:, 0]
        
        initial_n = node_inds.size(0)
        # inds = np.array([0] * resample_to_n)
        inds = np.random.choice(np.arange(initial_n), resample_to_n)
        return node_inds[inds].cuda(), edge_types[inds].cuda()    # torch.long



