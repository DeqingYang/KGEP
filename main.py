# coding: utf-8
from tqdm import tqdm
import json
from environment import environment
from agent import MultiAgents
import torch
import numpy as np
from torch import nn
from judge import Judge
from dataset import TrainDataset, TestDataset, get_dataloader
import random
import time
import pickle
from ipdb import set_trace
from evaluate import get_performance,ranklist_by_sorted
from time import time


class Conf:
    def __init__(self):
        self.dataset = "book"
        if self.dataset == "book":
            self.graph_num = 209081
            self.edge_type_num = 81 #39*2+2+1
            self.user2item = 78
            self.self_loop = 80
        if self.dataset=='douban':
            self.graph_num=196444
            self.edge_type_num = 14
            self.user2item = 0
            self.self_loop = 13
        if self.dataset=='music':
            self.graph_num =121216
            self.edge_type_num = 13 
            self.user2item = 10
            self.self_loop = 12
        self.dim = 64
        self.batch_size = 5000
        self.pre_train = False

        self.LR =1e-4 #learning rate
        self.EPOCH = 2000
        self.path_num = 4  #path number
        self.n_agents = 1  #agent number
        self.episode_len = 3 #path length

        self.sim_decay=0.008 #parameter for independence loss
        self.rel_decay=0.008 #parameter for relavance loss
        self.outdir='./weights/'+self.dataset+'/'
        self.ks= [1,3,5,10]  
        self.use_ind_loss = True #whether use independence loss
        self.use_rel_loss = True #whether use relavance loss
class Model(nn.Module):
    def __init__(self, conf):
        super(Model, self).__init__()
        with open('./data/' + conf.dataset + '/graph_map.json', 'r') as f:
            graph_map = json.loads(f.read())
        self.env = environment(graph_map, conf.self_loop, './data/' + conf.dataset + '/node2id.json')

        # initialize embedding
        if conf.pre_train:
            self.embeddings = torch.nn.Embedding(conf.graph_num, conf.dim,max_norm=0.1).cuda()
            self.edge_embeds = torch.nn.Embedding(conf.edge_type_num, conf.dim,max_norm=0.1).cuda()
            fr = open('./embed'+'/transe.pkl','rb')    
            inf = pickle.load(fr)     
            fr.close() 
            ent_emb,rel_emb = inf['weights']['entityEmbed'],inf['weights']['relationEmbed']
            self.embeddings.weight.data.copy_(torch.from_numpy(ent_emb))
            self.edge_embeds.weight.data.copy_(torch.from_numpy(rel_emb))
        else:
            self.embeddings = torch.nn.Embedding(conf.graph_num, conf.dim, max_norm=0.1).cuda()#len(self.env.new_map), 64)
            self.edge_embeds = torch.nn.Embedding(conf.edge_type_num, conf.dim, max_norm=0.1).cuda()
        self.multiagents = MultiAgents(self.env, self.embeddings, self.edge_embeds, user2item_idx=conf.user2item, n_agents=conf.n_agents, episode_len=conf.episode_len)
        self.judge = Judge(conf)

    def forward(self, batch_start_inds, batch_end_inds, batch_label):
        outputs_embs, outputs_ids, start_embeds, end_embeds,history_embs, relavance = self.multiagents(batch_start_inds, batch_end_inds)
        logit, loss = self.judge(outputs_embs, batch_label, start_embeds, end_embeds,history_embs,relavance)
        return logit, loss

class Optimize():
    def __init__(self, model, lr):
        self.model = model
        self.judge_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def judge_opt(self, loss):
        self.judge_optimizer.zero_grad()
        loss.backward()
        self.judge_optimizer.step()

def train_step(batch_start_inds, batch_end_inds, batch_label, model, opt):
    ''' One train step
    '''
    _, loss = model(batch_start_inds, batch_end_inds, batch_label)
    opt.judge_opt(loss)
    return loss

def train(conf, loader, model, opt):
    train_loss = 0
    loader = tqdm(loader)
    for batch in loader:
        if random.randint(1, 4000) == 6666:
            a = time.time()
            model.env.create_new_map_emb()
            print('create_new_map_emb', time.time() - a)
        batch_start_inds = batch[0][:, 0].repeat(conf.path_num, 1).permute(1, 0)
        batch_end_inds = batch[0][:, 1].repeat(conf.path_num, 1).permute(1, 0)
        batch_label = batch[1]
        loss = train_step(batch_start_inds, batch_end_inds, batch_label, model, opt)
        train_loss += float(loss.cpu())
    return train_loss

def test(conf, loader, model,test_users):
    Ks=conf.ks
    loader = tqdm(loader)
    rst = []
    logit_list = []
    total_user=test_users
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'map': np.zeros(len(Ks)),
              'hit':np.zeros(len(Ks)),
             'rr':0}

    for batch in loader:
        batch_start_inds = batch[0][:, 0].repeat(conf.path_num, 1).permute(1, 0)
        batch_end_inds = batch[0][:, 1].repeat(conf.path_num, 1).permute(1, 0)
        batch_label = batch[1]
        logit, _ = model(batch_start_inds, batch_end_inds, batch_label)#logit:[B,1]
        for i in range(logit.size()[0] // 50):#5 positive items and 45 negative items
            pos_items=batch[0][:, 1][(i * 50): ((i + 1) * 50)][:5].cpu().detach().tolist()
            test_items=batch[0][:, 1][(i * 50): ((i + 1) * 50)].cpu().detach().tolist()
            rating = logit[(i * 50): ((i + 1) * 50)][:, 1].cpu().detach().tolist()
            r,auc,rr=ranklist_by_sorted(pos_items, test_items, rating, Ks)
            result_oneuser=get_performance(pos_items, r, auc, Ks,rr)
            result['precision']+=result_oneuser['precision']
            result['recall']+=result_oneuser['recall']     
            result['ndcg']+=result_oneuser['ndcg']
            result['map']+=result_oneuser['map']
            result['rr']+=result_oneuser['rr']
            result['hit']+=result_oneuser['hit']
    result['precision']/=total_user
    result['recall']/=total_user  
    result['ndcg']/=total_user
    result['map']/=total_user 
    result['rr']/=total_user
    result['hit']/=total_user
    return result
          

if __name__ == '__main__':
    conf = Conf()
    # construct dataset
    data_train_fn = './data/' + conf.dataset + '/Interact_tuple_train.dat'
    data_test_fn = './data/' + conf.dataset + '/test_set.dat'
    node2id_fn = './data/' + conf.dataset + '/node2id.json'
    train_set = TrainDataset(data_train_fn, node2id_fn, 'train')
    test_set = TestDataset(data_test_fn, node2id_fn, 'test')
    test_users = test_set.test_users
    train_loader = get_dataloader(train_set, conf.batch_size, True)
    test_loader = get_dataloader(test_set, conf.batch_size, False)
    # construct model
    model = Model(conf)
    model = model.cuda()
    opt = Optimize(model, conf.LR)

    for epoch in range(conf.EPOCH):
        if epoch >=30:
            torch.save(model.state_dict(), conf.outdir + 'model_'  +'epoch_'+str(epoch)+ '.ckpt')
        if epoch % 10 == 7:
            conf.LR *= 0.9
            opt = Optimize(model, conf.LR)
        print('epoch:', epoch)
        
        train_s_t = time()
        train_loss = train(conf, train_loader, model, opt)
        train_e_t = time()
        
        test_s_t = time()
        test_rst = test(conf, test_loader, model,test_users)
        test_e_t = time()

        with open('result' + '.txt', 'a') as f:
            f.write(str(epoch)+'\t'+str(train_loss)+'\t'+str(train_e_t-train_s_t)+'\t'+str(test_e_t-test_s_t)+'\t'+str(test_rst.items())+'\n')


