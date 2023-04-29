import json
import os.path
import shutil
import time

import cv2
import matplotlib
import numpy as np
import pyproj
from matplotlib import pyplot as plt
from pyproj import Geod
from shapely import ops
from shapely.geometry import LineString, MultiPoint, Point
from shapely.geometry.multilinestring import MultiLineString
from shapely.ops import transform
from sklearn.metrics.pairwise import cosine_similarity
from dataset import Dataset

matplotlib.use('TkAgg')
import osmnx as ox
from argparse import ArgumentParser




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Metadate path')
    # parser.add_argument('-r', type=float, default=1000, help='Radius in meters')
    # parser.add_argument('-v', action='store_true', help='Option for visualization')
    # parser.add_argument('-s', default=None, type=str, help='Option for save')
    # parser.add_argument('-d', default=30, type=int, help='Distance between coordinates in meters')
    #
    opt = parser.parse_args()
    dataset = Dataset(opt.input, descriptor_type='radenovic_gldv1')

    # dataset1 = Dataset('datasets/Lviv49.8443931@24.0254815', descriptor_type='radenovic_gldv1')
    #
    coords = set([(m['lat'], m['lng']) for m in dataset.metadata(dataset.entries)])
    # coords1 = set([(m['lat'], m['lng']) for m in dataset1.metadata(dataset1.entries)])
    #
    # # pano = set([m['pano'] for m in dataset.metadata(dataset.entries)])
    # # pano1 = set([m['pano'] for m in dataset1.metadata(dataset1.entries)])
    #
    #
    #
    # print(len(coords.intersection(coords1)), len(coords), len(coords1))
    #
    # # print(len(coords.intersection(coords1)), len(pano.intersection(pano1)))
    #
    #
    # d = np.vstack(dataset.descriptor(dataset.entries))
    # d1 = np.vstack(dataset1.descriptor(dataset1.entries))
    # #
    # mat = cosine_similarity(d, d1)
    # #
    # col = np.argmax(mat, axis=-1)
    # #
    # res = np.isclose(np.max(mat, axis=-1), np.ones(len(mat)))
    #
    # idx = np.where(res)[0]
    # idx1 = col[res]
    #
    # e = np.asarray(dataset.entries)[idx]
    # e1 = np.asarray(dataset1.entries)[idx1]
    #
    # for i, j in zip(e, e1):
    #     im = np.asarray(dataset.image(i))
    #     im1 = np.asarray(dataset1.image(j))
    #
    #     if not np.allclose(im, im1):
    #         print(i, j)
    # # cv2.imshow('0', im)
    # # cv2.imshow('1', im1)
    # # cv2.waitKey()
    #
    # coords = np.asarray([(m['lat'], m['lng']) for m in dataset.metadata(e)])
    # coords1 = np.asarray([(m['lat'], m['lng']) for m in dataset1.metadata(e1)])
    #
    # geod = Geod(ellps='WGS84')
    #
    #
    # _, _, dists = geod.inv(coords[:, 0], coords[:, 1], coords1[:, 0], coords1[:, 1])

    pass
    # coords = {}
    # for entry in dataset.entries:
    #     m = dataset.metadata(entry)
    #     coords[(m['lat'], m['lng'])] = entry
    #
    # coords1 = {}
    # for entry in dataset1.entries:
    #     m = dataset1.metadata(entry)
    #     coords1[(m['lat'], m['lng'])] = entry
    #
    # keep = {v for k, v in coords.items() if k not in coords1}
    #
    # print(len(keep))

    # for entry in pano.intersection(pano1):
    #     shutil.rmtree(os.path.join(dataset.root, entry))
        # print(entry)




    g_city = ox.load_graphml('data/graph.graphml')
    fig, axis = ox.plot_graph(g_city, node_size=1, show=False, figsize=(12, 12), bgcolor="w")
    plt.scatter(*(list(zip(*list(coords)))[::-1]), c='b', marker='o', s=1)
    # plt.scatter(*(list(zip(*list(coords1)))[::-1]), c='r', marker='o', s=1)
    plt.show()

