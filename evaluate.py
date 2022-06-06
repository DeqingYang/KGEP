from metrics import *

import torch
import numpy as np
import heapq
from time import time
import random



def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = AUC(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in range(len(test_items)):
        item_score[test_items[i]] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(50, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    rank=0
    for i in K_max_item_score:
        rank+=1
        if i in user_pos_test:
            break
    auc = get_auc(item_score, user_pos_test)
    return r[:K_max], auc, 1/rank

def get_performance(user_pos_test, r, auc, Ks,rr):
    precision, recall, ndcg, mean_p,hit = [], [], [], [],[]
    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))
        mean_p.append(average_precision(r,K))
        hit.append(hit_at_k(r, K))
        
    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'map': np.array(mean_p), 'hit':np.array(hit), 'rr':rr}


