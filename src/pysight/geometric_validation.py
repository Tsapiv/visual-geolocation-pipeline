from typing import List, Iterable

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import euclidean_distances

from .common.camera import Camera


def filter_distant_cameras(cameras: Iterable[Camera], distance_thr: float):
    positions = np.asarray(list(map(lambda x: x.extrinsic.C, cameras)))

    graph = euclidean_distances(positions, positions) < distance_thr

    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    values, counts = np.unique(labels, return_counts=True)

    mode = values[np.argmax(counts)]

    return np.squeeze(np.where(labels == mode))
