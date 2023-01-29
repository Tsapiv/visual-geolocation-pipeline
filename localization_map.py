import re
from argparse import ArgumentParser
from typing import List

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances



def remove_duplication(array):
    filtered = []
    check_pool = set()
    for idx, el in enumerate(array):
        if el not in check_pool:
            check_pool.add(el)
            filtered.append(idx)
    return np.asarray(filtered)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', required=True, type=str, help='Prefix to *_features.npy and *_id.json')
    parser.add_argument('-k', required=False, type=int, default=3)
    parser.add_argument('-v', action='store_true', help='Option for visualization')

    args = parser.parse_args()

    features = np.squeeze(np.load(f'{args.input}-features.npy'))
    ids: List[str] = json.load(open(f'{args.input}-id.json'))

    g_city = ox.graph_from_place('Lviv', network_type='drive')
    print('Finish loading')

    identifiers = np.asarray([list(map(float, id_.split('/')[-1].split('_')[0].split(','))) for id_ in ids])
    nodes = np.asarray(ox.nearest_nodes(g_city, *(list(zip(*identifiers))[::-1])))

    order = np.argsort(nodes)

    n = 300
    # distance = np.squeeze(np.sqrt(np.sum(features - features[0], axis=-1) ** 2))
    similarity = 1-cosine_distances(features, [features[n]])
    # similarity = features @ features[n] / (np.linalg.norm(features, axis=-1) * np.linalg.norm(features[n]))

    similarity = similarity[order]
    nodes = nodes[order]

    nc = []
    ns = []
    prev = None
    ptr = 0
    print(np.max(similarity))
    for i, node in enumerate(g_city.nodes):
        if node == nodes[ptr]:
            ptr2 = ptr + 1
            while ptr2 < len(nodes) and nodes[ptr] == nodes[ptr2]:
                ptr2 += 1
            sim = np.max(similarity[ptr: ptr2])
            if np.isclose(sim, 1.0):
                # nc.append('orange')
                ns.append(sim)
            else:
                # nc.append('blue')
                ns.append(sim)
            ptr = min(ptr2, len(nodes) - 1)
        else:
            # nc.append('red')
            ns.append(np.min(similarity))


    tmp = np.asarray(ns)

    selected = np.argsort(tmp)[:-args.k]
    tmp[selected] = np.min(similarity)
    # print(np.sort(tmp)[::-1][:20])

    tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))

    tmp = tmp ** 3

    nc = [matplotlib.colors.to_hex(c) for c in plt.get_cmap('magma')(tmp)]

    ox.plot_graph(g_city, node_color=nc, node_size=tmp * 100, figsize=(12, 12), bgcolor="w")
