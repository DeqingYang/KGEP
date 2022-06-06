import json
from douban_preprocess import read_from_pickle, save_pickle
import collections
import numpy as np
import random

# with open('../data/book/graph_map.json', 'r') as f:
#     graph_map = json.loads(f.read())

# [Interact_tuple_train] = read_from_pickle('../data/book/Interact_tuple_train.dat')
# [Interact_tuple_test] = read_from_pickle('../data/book/Interact_tuple_test.dat')

# Interact_tuple_train[0]
# ['u0', 'i31625', [5, 20100831], ['i7379', 'i7716', 'i968', 'i8658', 'i9287', 'i39950', 'i36850', 'i39067', 'i8082']]

# Interact_tuple_test[0]
# ['u0', 'i15510', [5, 20130725], ['i20600', 'i15774', 'i29290', 'i20504', 'i29726', 'i12167', 'i17796', 'i38238', 'i4294']]

# graph_map['u']['u0']
# [[0, 'i31625'], [0, 'i2790'], [0, 'i12977']]
Interact_tuple_train = []
Interact_tuple_test = []

lines = open('../data/music/original/interaction.txt', 'r').readlines()
user_histroy = collections.defaultdict(list)
for l in lines:
    tmps = l.strip()
    inters = [int(i) for i in tmps.split(' ')]
    user_histroy[inters[0]] += inters[1:]

kg_np = np.loadtxt('../data/music/original/kg_final.txt', dtype=np.int32)
graph_map = {'a': collections.defaultdict(list), 'u': collections.defaultdict(list), 'i': collections.defaultdict(list)}
for line in kg_np:
    line = list(line)
    if line[0] < 19870:
        line[0] = 'i' + str(int(line[0]))
    else:
        line[0] = 'a' + str(int(line[0]))
    if line[2] < 19870:
        line[2] = 'i' + str(int(line[2]))
    else:
        line[2] = 'a' + str(int(line[2]))
    if line[0][0] == 'a' and line[2][0] == 'a':
        continue
    graph_map[line[0][0]][line[0]].append([int(line[1]), line[2]])
    graph_map[line[2][0]][line[2]].append([int(line[1])+5, line[0]])

Interact_tuple_train,Interact_tuple_test = [],[]

for u_id in user_histroy:
    random.shuffle(user_histroy[u_id])
    partition = int(len(user_histroy[u_id]) * 0.7)
    tmp = []
    for pos in user_histroy[u_id]:
        neg = []
        while len(neg) < 9:
            a = random.randint(0, 19870)
            if a not in user_histroy[u_id]:
                neg.append(a)
        tmp.append(['u' + str(u_id), 'i' + str(pos), [5, 0], ['i' + str(i) for i in neg]])
        if len(graph_map['u']['u' + str(u_id)])<partition:
            graph_map['u']['u' + str(u_id)].append([10,'i' + str(pos)])
            graph_map['i']['i' + str(pos)].append([11, 'u' + str(u_id)])
    Interact_tuple_train += tmp[: partition]
    Interact_tuple_test += tmp[partition:]


with open('../data/music/graph_map.json', 'w') as f:
    f.write(json.dumps(graph_map))


save_pickle([Interact_tuple_train], '../data/music/Interact_tuple_train.dat')

train_set=collections.defaultdict(dict)
for item in Interact_tuple_train:
    if train_set[item[0]].get('pos'):
        train_set[item[0]]['pos'].append(item[1])
        train_set[item[0]]['neg'].extend(item[3])
    else:
        train_set[item[0]]['pos']=[]
        train_set[item[0]]['pos'].append(item[1])
        train_set[item[0]]['neg']=[]
        train_set[item[0]]['neg'].extend(item[3])
        
test_set=collections.defaultdict(dict)
for item in Interact_tuple_test:
    if test_set[item[0]].get('pos'):
        test_set[item[0]]['pos'].append(item[1])
        test_set[item[0]]['neg'].extend(item[3])
    else:
        test_set[item[0]]['pos']=[]
        test_set[item[0]]['pos'].append(item[1])
        test_set[item[0]]['neg']=[]
        test_set[item[0]]['neg'].extend(item[3])

total=[]
for i in range(0,19870):
    total.append('i'+str(i))
    
test_new=collections.defaultdict(dict)

for i in Interact_tuple_test:
    if test_new[i[0]].get('pos'):
        if len(test_new[i[0]]['pos'])<5:
            test_new[i[0]]['pos'].append(i[1])
            excep = train_set[i[0]]['pos']+train_set[i[0]]['neg']+test_set[i[0]]['pos']
            left=set(total)-set(excep)
            neg=random.sample(left,9)
            test_new[i[0]]['neg'].extend(neg)
    else:
        test_new[i[0]]['pos']=[]
        test_new[i[0]]['pos'].append(i[1])
        test_new[i[0]]['neg']=[]
        excep = train_set[i[0]]['pos']+train_set[i[0]]['neg']+test_set[i[0]]['pos']
        left=set(total)-set(excep)
        neg=random.sample(left,9)
        test_new[i[0]]['neg'].extend(neg)

test_set_music=collections.defaultdict(dict)#只包含5个正样本和45个负样本
for user in test_new:
    if len(test_new[user]['pos'])==5 and len(test_new[user]['neg'])==45:
        test_set_music[user]=test_new[user]
    
save_pickle([test_set_music], '../data/music/test_set.dat')
print('done')