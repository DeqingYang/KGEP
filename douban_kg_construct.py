import pickle
from douban_preprocess import construct_id_dict, save_pickle, read_from_pickle
import math
from ipdb import set_trace
import random

if __name__ == '__main__':
    [comments] = read_from_pickle('../data/original/douban/comments.dat')
    [actors_dict, actors_table] = read_from_pickle('../data/douban/oroginal/actors.dat')
    [directors_dict, directors_table] = read_from_pickle('../data/douban/original/directors.dat')
    [writers_dict, writers_table] = read_from_pickle('../data/douban/original/writers.dat')
    [type_dict, type_table] = read_from_pickle('../data/douban/original/type.dat')

    #七种边的类型：
    # edge_dict = {'Interact': 0, 'IsActorOf': 1, 'IsDirectorOf': 2,'IsWriterOf': 3, 'IsTypeOf': 4, 'InterestIn': 5,'HasLabel,attribution':6}

    #三种节点集合: user_list, item_list, attr_list
    user_number = comments[-1][0] + 1
    user_list = ['u' + str(i) for i in range(user_number)]

    item_number = actors_table[-1][0] + 1
    item_list = ['i' + str(i) for i in range(item_number)]

    label_list = [v for line in comments for v in line[4]]
    attr_old_list = list(actors_dict.keys()) + list(directors_dict.keys()) + list(writers_dict.keys()) + list(type_dict.keys()) + label_list
    attr_dict, attr_number = construct_id_dict(attr_old_list)  #'安迪·巴克利': 16289
    attr_list = ['a' + str(a) for a in range(len(attr_dict))]

    # 拆分comments
    comments_kg, comments_train, comments_test, tmp = [], [], [], []
    cur_user = comments[0][0]
    comments.append(comments[-1])
    comments[-1][0] += 1
    for line in comments:
        if line[2] < 4:
            continue
        if tmp != [] and cur_user != line[0]:
            tmp.sort(key=lambda line: line[3])
            neg_sample = list(set([item_list[random.randint(0, len(item_list) - 1)] for _ in range(len(tmp) * 18)]))
            partition1 = math.ceil(len(tmp) * 0.7)
            partition2 = math.ceil(len(tmp) * 0.85)
            tmp = [tmp[i] + [neg_sample[(i * 9): ((i + 1) * 9)]] for i in range(len(tmp))]
            # comments_kg += [line[:-1] for line in tmp[:partition1]]
            # comments_train += tmp[partition1: partition2]
            # comments_test += tmp[partition2:]
            comments_kg += [line[:-1] for line in tmp[:partition1]]
            comments_train += tmp[:partition1]
            comments_test += tmp[partition1:]
            tmp = []
        else:
            tmp.append(line[:])
        cur_user = line[0]
    print(len(comments_kg), len(comments_train), len(comments_test))


    #七种三元组集合
    Interact_tuple_kg, Interact_tuple_train, Interact_tuple_test, IsActorOf_tuple, IsDirectorOf_tuple, IsWriterOf_tuple, IsTypeOf_tuple, InterestIn_tuple, HasLabel_tuple = [], [], [], [], [], [], [], [], []

    actors_reverse_dict = dict(zip(actors_dict.values(), actors_dict.keys())) #17527: '多部未華子'
    directors_reverse_dict = dict(zip(directors_dict.values(), directors_dict.keys()))
    writers_reverse_dict = dict(zip(writers_dict.values(), writers_dict.keys()))
    type_reverse_dict = dict(zip(type_dict.values(), type_dict.keys()))

    # [4963, 36897, 4, 20170315, ['宗教', '心理', '悬疑', '惊悚', '意外结局', '反转']]
    Interact_tuple_test = [['u' + str(line[0]), 'i' + str(line[1]), [line[2], line[3]], line[5]] for line in comments_test]
    Interact_tuple_train = [['u' + str(line[0]), 'i' + str(line[1]), [line[2], line[3]], line[5]] for line in comments_train]

    for line in comments_kg:
        Interact_tuple_kg.append(['u' + str(line[0]), 0, 'i' + str(line[1]), [line[2], line[3]]])
        for v in line[4]:
            InterestIn_tuple.append(['u' + str(line[0]), 5, 'a' + str(attr_dict[v])])
            HasLabel_tuple.append(['i' + str(line[1]), 6, 'a' + str(attr_dict[v])])

    IsActorOf_tuple = [['a' + str(attr_dict[actors_reverse_dict[actor]]), 1, 'i' + str(line[0])] for line in actors_table for actor in line[1]]
    IsDirectorOf_tuple = [['a' + str(attr_dict[directors_reverse_dict[director]]), 2, 'i' + str(line[0])] for line in directors_table for director in line[1]]
    IsWriterOf_tuple = [['a' + str(attr_dict[writers_reverse_dict[writers]]), 3, 'i' + str(line[0])] for line in writers_table for writers in line[1]]
    IsTypeOf_tuple = [['a' + str(attr_dict[type_reverse_dict[type_]]), 4, 'i' + str(line[0])] for line in type_table for type_ in line[1]]

    set_trace()

    save_pickle([user_list, item_list, attr_list],'../data/douban/original/kg_node.dat')
    save_pickle([Interact_tuple_kg, InterestIn_tuple, HasLabel_tuple, IsActorOf_tuple, IsDirectorOf_tuple, IsWriterOf_tuple, IsTypeOf_tuple],'../data/douban/original/kg_edge.dat')
    save_pickle([attr_dict], '../data/douban/original/attr_dict.dat')
    # save_pickle([Interact_tuple_test], './data/douban/Interact_tuple_test.dat')
    save_pickle([Interact_tuple_train], '../data/douban/Interact_tuple_train.dat')
    
    print('The number of node is ',len(user_list + item_list + attr_list))
    print('user:', len(user_list))
    print('item:', len(item_list))
    print('attr:', len(attr_list))
    print('eg:', user_list[0], item_list[0], attr_list[0],'\n')
    print('The number of edge is ',len(Interact_tuple_kg + InterestIn_tuple + HasLabel_tuple + IsActorOf_tuple + IsDirectorOf_tuple + IsWriterOf_tuple + IsTypeOf_tuple))
    print('eg:')
    print(Interact_tuple_kg[0])
    print(Interact_tuple_train[0])
    print(Interact_tuple_test[0])
    print(IsActorOf_tuple[0])
    print(IsDirectorOf_tuple[0])
    print(IsWriterOf_tuple[0])
    print(IsTypeOf_tuple[0])
    print(InterestIn_tuple[0])
    print(HasLabel_tuple[0])
    
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
for i in range(0,41785):
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

test_set_douban=collections.defaultdict(dict)#只包含5个正样本和45个负样本
for user in test_new:
    if len(test_new[user]['pos'])==5 and len(test_new[user]['neg'])==45:
        test_set_douban[user]=test_new[user]

save_pickle([test_set_douban], '../data/douban/test_set.dat')