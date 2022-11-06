import re

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
import json
import cv2


def remove_duplication(array):
    filtered = []
    check_pool = set()
    for idx, el in enumerate(array):
        if el not in check_pool:
            check_pool.add(el)
            filtered.append(idx)
    return np.asarray(filtered)


if __name__ == '__main__':
    features = np.squeeze(np.load('data/region_filtered_ViT-L14_features.npy'))
    filenames = json.load(open('data/region_filtered_ViT-L14_filenames.json'))
    metadata = json.load(open('data/region1_metadata.json'))
    g_city = ox.graph_from_place('Lviv', network_type='drive')
    print('Finish loading')

    metadata = {node['pano_id']: ox.nearest_nodes(g_city, node['location']['lng'], node['location']['lat']) for node in
                metadata}
    n = 11
    # distance = np.squeeze(np.sqrt(np.sum(features - features[0], axis=-1) ** 2))
    distance = features @ features[n] / (np.linalg.norm(features, axis=-1) * np.linalg.norm(features[n]))

    indexes = np.argsort(distance)
    distance = distance[indexes]

    nodes = np.asarray([metadata[re.search(r".+/(.+)_\d{1,3}\.png", filename).group(1)] for filename in
                        np.asarray(filenames)[indexes]])

    # _, idx = np.unique(nodes[::-1], return_index=True)
    # idx = (len(idx) - np.sort(idx)[::-1])[:10]
    # idx = remove_duplication(nodes[::-1])
    # idx = (len(idx) - idx[::-1])[-100:]
    # nodes = nodes[idx]
    # distance = distance[idx]

    max_dist = np.max(distance)
    min_dist = np.min(distance)

    nc = []
    ns = []
    for i, node in enumerate(g_city.nodes):
        idx = np.where(nodes == node)[0]

        if len(idx) > 0:
            ns.append(np.max(distance[idx]))
        else:
            ns.append(min_dist)

    tmp = np.asarray(ns)

    tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))

    tmp = tmp ** 4

    nc = [matplotlib.colors.to_hex(c) for c in plt.get_cmap('magma')(tmp)]
    print(nc)
    print(tmp)

    ox.plot_graph(g_city, node_color=nc, node_size=tmp * 100, figsize=(12, 12), bgcolor="w")
