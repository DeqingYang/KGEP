# coding: utf-8

import torch
import numpy as np
from math import ceil
from torch import nn
from ipdb import set_trace
import time
import random

class gru(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(gru, self).__init__()
        self.cell = nn.GRUCell(input_size, hidden_size)
        self.first_flag = True
    
    def forward(self, x,h):
        if self.first_flag:
            self.h = self.cell(x,h)
        else:
            self.h, self.c = self.cell(x, (self.h, self.c))
        return self.h



class Agent(nn.Module):
    ''' Agent for exploring path. '''

    def __init__(self, env, embeddings, edge_embeds, episode_len=10, \
                    embed_dim=64, lstm_hidden=64, dropout=0.5):
        super(Agent, self).__init__()
        self.env = env
        self.embeddings = embeddings.cuda()  # torch.nn.Embedding
        self.edge_embeds = edge_embeds.cuda()
        self.episode_len = episode_len
        self.embed_dim = embed_dim

        self.gru_cell = gru(input_size=embed_dim, hidden_size=lstm_hidden)

        self.fc_layer = nn.Sequential(
            nn.Linear(lstm_hidden, out_features=lstm_hidden),
            nn.ReLU(),
            nn.Linear(lstm_hidden, embed_dim),
        )

    def forward(self, start_inds, end_inds, user2item_idx):
        ''' Generate an episode.
        start_inds/ end_inds: torch.long tensor with size [1]
        '''
        start_embeds = self.embeddings(start_inds)
        end_embeds = self.embeddings(end_inds)
        
        relation_embeds= self.edge_embeds(torch.full((start_inds.shape[0],),user2item_idx).cuda())
        h=start_embeds+end_embeds

        B = start_inds.size(0)
        query = torch.zeros([B, self.embed_dim]).cuda()+ start_embeds
        
        history_path=[start_embeds,relation_embeds,end_embeds]
        
        relavance = torch.zeros([B, 1]).cuda()
        
        current_inds, current_embeds, prev_edge_embeds,h, rel = self.sub_forward(h, query, start_inds,history_path)

        output_embeds = [start_embeds, current_embeds]
        output_inds = [start_inds, current_inds]

        relavance += rel
        for _ in range(self.episode_len - 2):
            # query= torch.mul(prev_edge_embeds, current_embeds)
            query = prev_edge_embeds + current_embeds + start_embeds + end_embeds
            current_inds, current_embeds, prev_edge_embeds,h, rel = self.sub_forward(h, query, current_inds,history_path)
            relavance += rel
            history_path.append(prev_edge_embeds)
            history_path.append(current_embeds)
            output_embeds.append(current_embeds)
            output_inds.append(current_inds)
        lists = np.array([ts.tolist() for ts in output_inds + [end_inds]])
        # np.save('./rst/' + str(random.randint(1, 999)) + '.npy', lists)
        return output_embeds, output_inds, start_embeds, end_embeds,history_path,relavance


    def sub_forward(self, h, query, current_inds,history_path):
        ''' Perform a sub forward step.
        h:              previous action and current node embeddings
        query:          user, item, current node and edge 
        current_inds:   index indicating the current node.
        history_path:   previous path, including previous edge embeddings and previous node embeddings
        '''
        h = self.gru_cell(query,h)
        # mlp_policy
        x = self.fc_layer(h)
        x = x.unsqueeze(1)

        node_inds, edge_types = self.env.get_action_new(current_inds) 
        cand_edge_embeds = self.edge_embeds(edge_types)  # [B, n, hidden]
        candidates = self.embeddings(node_inds)    # [B, n, hidden]
        actions = cand_edge_embeds + candidates
        pro = (actions*x).sum(2,True).squeeze(-1)
        pro = pro.float()
        pro[torch.isnan(pro)] = 0
        pro[torch.isinf(pro)] = 1
        d=torch.nn.functional.softmax(pro,dim=1) #probability of the actions
        tmp_inds = d.multinomial(1).view(-1)    # sample
        row_inds = np.arange(tmp_inds.size(0))
        # get next node indices, embeddings and relavance score of the choosed action
        next_inds = node_inds[row_inds, tmp_inds]
        next_embeds = candidates[row_inds, tmp_inds]
        edge_embeds = cand_edge_embeds[row_inds, tmp_inds]
        relevance = ((next_embeds+edge_embeds)*x.squeeze(1)).sum(1).unsqueeze(-1)
        return next_inds, next_embeds, edge_embeds,h, relevance



class MultiAgents(nn.Module):
    ''' Aggregated multiple agents. '''
    def __init__(self, env, embeddings, edge_embeds,  user2item_idx=0, n_agents=4, \
                    episode_len=10, embed_dim=64, lstm_hidden=64, dropout=0.5):
        super(MultiAgents, self).__init__()
        self.agents = nn.ModuleList()
        self.user2item_idx = user2item_idx
        for _ in range(n_agents):
            self.agents.append(Agent(env, embeddings, edge_embeds, \
                        episode_len, embed_dim, lstm_hidden, dropout))


    def __call__(self, batch_start_inds, batch_end_inds):
        ''' Generate an episode for multiple agents 
        batch_start/end_inds should be integer Tensors of shape [B, n_start/end]
        '''
        # 500, 4
        B, n = batch_start_inds.size()
        outputs_embs, outputs_ids ,history_embs= [], [],[]

        batch_start_inds = batch_start_inds.reshape(-1)
        batch_end_inds = batch_end_inds.reshape(-1)


        for ag in self.agents:
            output_embeds, output_inds, start_embeds, end_embeds ,history_path ,relavance= ag(batch_start_inds, batch_end_inds,self.user2item_idx)
            #output_embeds :[n,B*path_num,embed_dim], n indicates the number of nodes in the path
            #history path :[k,B*path_num,embed_dim], k indicates the number of relations and nodes in the path
            stacked_embeds = torch.stack(output_embeds).transpose(0, 1)#[B*path_num,episode_len,embed_dim]
            stacked_inds = torch.stack(output_inds).transpose(0, 1)

            stacked_embeds = stacked_embeds.view(B, n, stacked_embeds.size(1),stacked_embeds.size(2))#[B,num_path,episode_len,embed_dim]
            stacked_inds = stacked_inds.view(B, n, stacked_inds.size(1))

            outputs_embs.append(stacked_embeds)
            outputs_ids.append(stacked_inds)
            
            stacked_history_emb = torch.stack(history_path).transpose(0,1)
            stacked_history_emb = stacked_history_emb.view(B, n, stacked_history_emb.size(1),stacked_history_emb.size(2))
            #stacked_history_emb: [B,num_path,episode_len,embed_dim]
            history_embs.append(stacked_history_emb)
        
        outputs_embs = torch.stack(outputs_embs).transpose(0, 1)
        outputs_ids = torch.stack(outputs_ids).transpose(0, 1)
        history_embs =torch.stack(history_embs).transpose(0,1) #[B,num_agent,num_path,episode_len,embed_dim]
        return outputs_embs, outputs_ids, start_embeds, end_embeds,history_embs,relavance

