import json
import re
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from pyproj import Geod
from sklearn.metrics.pairwise import cosine_distances


def map_k(y_true: np.ndarray, k: int):
    assert len(y_true[0]) >= k
    y_true = y_true[:, :k]
    denom = np.arange(1, k + 1)
    return np.mean(np.cumsum(y_true, axis=-1) * y_true / denom, axis=-1)


def order_k(distances: np.ndarray, k: int):
    index_array1 = np.argpartition(distances, kth=k, axis=-1)[:, :k]
    distances = np.take_along_axis(distances, index_array1, axis=-1)
    index_array2 = np.argsort(distances, axis=-1)
    return np.take_along_axis(index_array1, index_array2,
                              axis=-1)


def order_by_distance(lats, lngs):
    geod = Geod(ellps='WGS84')

    X, Y = np.mgrid[0:len(lats):1, 0:len(lats):1]

    _, _, dists = geod.inv(lngs[X.ravel()], lats[X.ravel()], lngs[Y.ravel()], lats[Y.ravel()])

    dist_mat = np.zeros((len(lats), len(lats)))

    dist_mat[X.ravel(), Y.ravel()] = dists
    return dist_mat


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', required=True, type=str, help='Prefix to *_features.npy and *_id.json')
    parser.add_argument('-k', required=False, type=int, default=3)
    parser.add_argument('-v', action='store_true', help='Option for visualization')

    args = parser.parse_args()

    features = np.squeeze(np.load(f'{args.input}-features.npy'))
    ids = json.load(open(f'{args.input}-id.json'))
    identifiers = [re.search(r".+/(.+)_\d{1,3}\.(png|jpeg|jpg)", id_).group(1) for id_ in ids]
    coords = np.asarray(list(map(lambda x: tuple(map(float, x.split('@'))), identifiers)))

    dist_mat = order_by_distance(coords[:, 0], coords[:, 1])

    _, indices = np.unique(identifiers, return_inverse=True)
    similarity_ordering = order_k(cosine_distances(features, features), args.k + 1)

    geo_ordering = order_k(dist_mat, 24)

    indices = indices[similarity_ordering]

    bitmap = []
    for i in range(len(similarity_ordering)):
        bitmap.append(np.in1d(similarity_ordering[i], geo_ordering[i]))
    bitmap = np.asarray(bitmap)
    scores = map_k(bitmap, args.k)

    print(f'Mean: {np.mean(scores)}')
    print(f'Median: {np.median(scores)}')
    print(f'Std: {np.std(scores)}')

    if args.v:
        plt.hist(scores)
        plt.show()
