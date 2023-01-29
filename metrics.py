import json
import re
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances


def map_k(y_true: np.ndarray, k: int):
    assert len(y_true[0]) >= k
    y_true = y_true[:, :k]
    denom = np.arange(1, k + 1)
    return np.mean(np.cumsum(y_true, axis=-1) * y_true / denom, axis=-1)


def order_k(descriptors: np.ndarray, k: int):
    distances = cosine_distances(descriptors, descriptors)
    index_array1 = np.argpartition(distances, kth=k, axis=-1)[:, :k]
    distances = np.take_along_axis(distances, index_array1, axis=-1)
    index_array2 = np.argsort(distances, axis=-1)
    return np.take_along_axis(index_array1, index_array2,
                              axis=-1)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', required=True, type=str, help='Prefix to *_features.npy and *_id.json')
    parser.add_argument('-k', required=False, type=int, default=3)
    parser.add_argument('-v', action='store_true', help='Option for visualization')

    args = parser.parse_args()

    features = np.squeeze(np.load(f'{args.input}-features.npy'))
    ids = json.load(open(f'{args.input}-id.json'))
    identifiers = np.asarray([re.search(r".+/(.+)_\d{1,3}\.(png|jpeg|jpg)", id_).group(1) for id_ in ids])
    _, indices = np.unique(identifiers, return_inverse=True)
    similarity_ordering = order_k(features, args.k + 1)

    indices = indices[similarity_ordering]

    bitmap = indices[:, 1:] == indices[:, 0, None]
    scores = map_k(bitmap, args.k)

    print(f'Mean: {np.mean(scores)}')
    print(f'Median: {np.median(scores)}')
    print(f'Std: {np.std(scores)}')

    if args.v:
        plt.hist(scores)
        plt.show()




