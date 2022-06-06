# coding: utf-8

import json
from douban_preprocess import read_from_pickle
from collections import defaultdict

if __name__ == '__main__':
    node_fn = '../data/douban/original/kg_node.dat'
    edge_fn = '../data/douban/original/kg_edge.dat'
    save_to_fn = '../data/douban/original/graph_map.json'

    user_list, item_list, attr_list = read_from_pickle(node_fn)
    Interact_tuple, InterestIn_tuple, \
        HasLabel_tuple, IsActorOf_tuple, \
            IsDirectorOf_tuple, IsWriterOf_tuple, \
                IsTypeOf_tuple      = read_from_pickle(edge_fn)
    # # 从attribute字符串映射到id
    # [attr_dict] = read_from_pickle('../data/douban/attr_dict.dat')
    # print(attr_dict)

    # 根据interact_tuple产生训练集，评分大于3分标为1.
    dataset = [[item[0], item[2], int(item[3][0] > 3)] for item in Interact_tuple]
    print('dataset sample:', dataset[:10])

    # 按照u->{a, i}, a->{i, u}, i->{u, a}生成映射表
    u_map, a_map, i_map = defaultdict(list), defaultdict(list), defaultdict(list)

    for user, edge_type, item, _ in Interact_tuple:
        u_map[user].append((edge_type, item))
        # 额外加上item到user的逆向边，编号12
#         i_map[item].append((12, user))

    for user, edge_type, attr in InterestIn_tuple:
        u_map[user].append((edge_type, attr))
        # 额外加上从attribute到user的逆向边，编号7
#         a_map[attr].append((7, user))
    
    for item, edge_type, attr in HasLabel_tuple:
        i_map[item].append((edge_type, attr))

    for attr, edge_type, item in IsActorOf_tuple + IsDirectorOf_tuple + IsWriterOf_tuple + IsTypeOf_tuple:
        a_map[attr].append((edge_type, item))
        # 额外加上逆向边，放弃haslabel，编号8，9，10，11
#         i_map[item].append((12 - edge_type, attr))

    maps = {'u': u_map, 'a': a_map, 'i': i_map}

    with open(save_to_fn, 'w') as f:
        f.write(json.dumps(maps))

    with open('../data/douban/original/graph_map.json', 'r') as f:
        graph_map = json.loads(f.read())

    M = {'u': 0, 'i': 0, 'a': 0}
    for v in graph_map.values():
        for kk, vv in v.items():
            M[kk[0]] = max(M[kk[0]], int(kk[1:]))
            for _, vvv in vv:
                M[vvv[0]] = max(M[vvv[0]], int(vvv[1:]))
    node2id = {}
    idx = 0
    for u in range(0, M['u'] + 1):
        node2id['u' + str(u)] = idx
        idx += 1
    for i in range(0, M['i'] + 1):
        node2id['i' + str(i)] = idx
        idx += 1
    for a in range(0, M['a'] + 1):
        node2id['a' + str(a)] = idx
        idx += 1
    with open('../data/douban/original/node2id.json', 'w') as f:
        f.write(json.dumps(node2id))
    print(M)