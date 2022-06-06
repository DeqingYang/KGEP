
import numpy as np
import torch
import torch.nn as nn
import random
from ipdb import set_trace
from torch.autograd import Variable
from itertools import combinations


class Judge(nn.Module):
    def __init__(self, conf):
        super(Judge, self).__init__()
        num1, num2 = 64, 64
        self.fc2 = nn.Sequential(nn.Linear(64 * 4, 111), nn.ReLU(), nn.Linear(111, 64)).cuda()
        self.fc3 = nn.Sequential(nn.Linear(64 * 2, 1)).cuda()
        self.MLP = nn.Sequential(nn.Linear(64 * 3, 1024), nn.ReLU(), nn.Linear(1024, 1)).cuda()
        self.EPOCH = conf.EPOCH
        self.LR = conf.LR
        self.batch_size = conf.batch_size
        self.dim = conf.dim
        self.path_num=conf.path_num
        self.sim_decay=conf.sim_decay
        self.temperature = 0.2
        self.use_ind_loss = conf.use_ind_loss
        self.use_rel_loss = conf.use_rel_loss
        self.rel_decay = conf.rel_decay

    def CosineSimilarity(self,matrix_3d):
        A=matrix_3d
        prod = torch.matmul(A, A.permute(0,2,1))
        norm = torch.norm(A,p=2,dim=2,keepdim=True)
        cos = prod.div(torch.matmul(norm,norm.permute(0,2,1)))
        b=torch.triu(cos, diagonal=1)
        b=b**2
        return b.sum()

    def p_score(self,matrix_3d):
        A=matrix_3d
        prod = torch.matmul(A, A.permute(0,2,1))
        norm = torch.norm(A,p=2,dim=2,keepdim=True)
        cos = prod.div(torch.matmul(norm,norm.permute(0,2,1)))
        cos = 1 - cos
        b=torch.triu(cos, diagonal=1)
        s = b.sum(2).sum(1).unsqueeze(-1)
        ild = s/(self.choose_num*(self.choose_num-1)/2)
        ild_loss = (-torch.log(ild/2+1e-8)).sum()/A.size(0)
        return  ild,ild_loss#[batchsize,1],num

    def MutualInformation(self,disen_3d):
        #disen_3d: [B,n_path,embed_dim]
        disen_T = disen_3d

        normalized_disen_T = disen_T / disen_T.norm(dim=2, keepdim=True)

        pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=2,keepdim=True)
        ttl_scores = torch.sum(torch.exp(torch.matmul(normalized_disen_T, normalized_disen_T.permute(0,2,1))/self.temperature), dim=2,keepdim=True)

        pos_scores = torch.exp(pos_scores / self.temperature)
        mi_score = - torch.sum(torch.log(pos_scores / ttl_scores),dim=1)  #[bacthsize,1]
        total = mi_score.sum()/disen_3d.size(0)
        return mi_score,total
    
    def forward(self, paths_emb, label, start_embeds, end_embeds,history_embs,relavance):
        return self.forward_judge_and_agents1(paths_emb, label, start_embeds, end_embeds,history_embs,relavance)



    def forward_judge_and_agents1(self, paths_emb, label, start_embeds, end_embeds,history_embs,relavance):
        #path_emb: [B,num_agents,n_path,episod_len,embed_dim]
        #label:[B*n_path]
        #start_embeds/end_embeds:[B*n_path,embed_dim]
        #history_embs:[B,num_agents,n_path,episod_len,embed_dim]
        #relavance:[B*n_path,1]
        paths_emb2=paths_emb.squeeze(1)#[B,n_path,episod_len,embed_dim]
        paths_emb2=torch.mean(paths_emb2,2)#[B,n_path,embed_dim]
        num_path=paths_emb2.size()[1]
        
        start_embeds = start_embeds.reshape((paths_emb.size(0), -1, 1, self.dim))[:, 0, 0, :] #[B,n_path,1,embed_dim]--->[B,embed_dim]
        end_embeds = end_embeds.reshape((paths_emb.size(0), -1, 1, self.dim))[:, 0, 0, :]       
        history_embs=history_embs[:,:,:,1:,:]
        path_emb=[]
        for i in range(0,history_embs.size(3),2):
            path_emb.append(history_embs[:,:,:,i,:]*history_embs[:,:,:,i+1,:]) #B,num_agents,n_path,embed_dim
        path_emb = torch.stack(path_emb).permute(1,2,3,0,4)#B,num_agents,n_path,episode_len,embed_dim
        path_emb = torch.mean(torch.mean(path_emb,1),2)#B,num_path,embed_dim
        relavance = relavance.reshape(paths_emb2.size(0), paths_emb2.size(1), -1)

        if self.use_ind_loss:
             _, loss_ind= self.MutualInformation(path_emb)
        user_emb = start_embeds.reshape((paths_emb.size(0), -1, self.dim)) #B,path_num,embed_dim
        x = torch.exp((path_emb*user_emb).sum(2,True))
        y = x.sum(1,True)
        path_attn = x/y #B,num_path,1
        path_emb = torch.mean(path_emb*path_attn,1,True).squeeze(1)
        loss_func = nn.CrossEntropyLoss()          
        out = self.MLP(torch.cat([start_embeds, path_emb, end_embeds], dim=1)).squeeze(-1)  # [B,1]
        logit = torch.sigmoid(out).reshape((-1, 1))
        p = torch.cat((logit, 1 - logit), 1)
        loss_logit = loss_func(p, label)

        if self.use_rel_loss and self.use_ind_loss:
            loss_cor = self.sim_decay * loss_ind
            rel = relavance.mean(dim=1)
            loss_rel =self.rel_decay * torch.mean(-torch.log(torch.sigmoid(rel)))
            loss=loss_logit+loss_cor+loss_rel
        elif self.use_rel_loss:
            rel = relavance.mean(dim=1)
            loss_rel =self.rel_decay * torch.mean(-torch.log(torch.sigmoid(rel)))
            loss=loss_logit+loss_rel
        elif self.use_ind_loss:
            loss_cor = self.sim_decay * loss_ind
            loss=loss_logit+loss_cor
        else:
            loss=loss_logit
        return p, loss

 