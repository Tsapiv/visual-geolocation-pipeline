from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from pyproj import Geod
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.model_selection import train_test_split

from camera import CameraMetadata
from dataset import Dataset
from geoutils import extrinsic_from_metadata


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
    parser.add_argument('--input', required=True, type=str, help='Path to train dataset')
    parser.add_argument('--input2', default=None, type=str, help='Path to test dataset')
    parser.add_argument('--descriptor_type', required=True, type=str, help='Prefix of descriptor filename')
    parser.add_argument('-k', required=False, type=int, default=3, help='k in map@k')
    parser.add_argument('-k2', required=False, type=int, default=23, help='Sets k closest in the trainset by distance between cameras')
    parser.add_argument('-s', '--split', type=float, default=0.2, help='Test size split')

    opt = parser.parse_args()

    if opt.input2 is None:
        dataset = Dataset(root=opt.input, descriptor_type=opt.descriptor_type)

        train_entries, test_entries = train_test_split(dataset.entries, test_size=opt.split)

        trainset = dataset.get_subset(train_entries)
        testset = dataset.get_subset(test_entries)
    else:
        trainset = Dataset(root=opt.input, descriptor_type=opt.descriptor_type)
        testset = Dataset(root=opt.input2, descriptor_type=opt.descriptor_type)

        train_entries, test_entries = trainset.entries, testset.entries


    train_coords = np.asarray([extrinsic_from_metadata(CameraMetadata.from_kwargs(**m)).C for m in trainset.metadata(train_entries)])
    test_coords = np.asarray([extrinsic_from_metadata(CameraMetadata.from_kwargs(**m)).C for m in testset.metadata(test_entries)])

    train_features = np.concatenate([features.reshape(1, -1) for features in trainset.descriptor(train_entries)])
    test_features = np.concatenate([features.reshape(1, -1) for features in testset.descriptor(test_entries)])

    distances = euclidean_distances(test_coords, train_coords)
    similarity = cosine_distances(test_features, train_features)

    distance_ordering = order_k(distances, opt.k2)
    similarity_ordering = order_k(similarity, opt.k)

    bitmap = []
    for i in range(len(similarity_ordering)):
        bitmap.append(np.in1d(similarity_ordering[i], distance_ordering[i]))
    bitmap = np.asarray(bitmap)
    scores = map_k(bitmap, opt.k)

    print(f'Mean: {np.mean(scores)}')
    print(f'Median: {np.median(scores)}')
    print(f'Std: {np.std(scores)}')

    plt.hist(scores)
    plt.show()
