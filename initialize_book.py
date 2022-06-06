# coding: utf-8

import json
from douban_preprocess import read_from_pickle
from collections import defaultdict

if __name__ == '__main__':
    with open('../data/book/graph_map.json', 'r') as f:
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
    with open('../data/book/node2id.json', 'w') as f:
        f.write(json.dumps(node2id))
    print(M)