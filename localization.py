import json
from argparse import ArgumentParser
from typing import List, Union

import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances, haversine_distances, manhattan_distances, euclidean_distances

from gpu_re_ranking.gnn_reranking import gnn_reranking

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', required=True, type=str, help='Prefix to *_features.npy and *_id.json')
    parser.add_argument('-k', required=False, type=int, default=3)
    parser.add_argument('--rerank', action='store_true', help='Option for reranking')
    parser.add_argument('-v', action='store_true', help='Option for visualization')
    parser.add_argument('-q', default='4000', type=str, help='Name/number of query picture')
    parser.add_argument('--pca_dim', default=None, type=int, help='Number of PCA dimensions')

    opt = parser.parse_args()
    ids: List[str] = json.load(open(f'{opt.input}-id.json'))

    k = opt.k + 1
    try:
        n = int(opt.q)
    except ValueError:
        n = list(map(lambda x: x.split('/')[-1], ids)).index(opt.q.split('/')[-1])

    gallery_features = np.squeeze(np.load(f'{opt.input}-features.npy'))

    query_feature = gallery_features[n].reshape(1, -1)
    # query_feature = np.load('/home/tsapiv/Documents/diploma/deep-visual-geo-localization-benchmark/test/default/2023-02-26_15-20-36/radenovic_gldv1-query-features.npy').reshape(1, -1)

    pca = None
    if opt.pca_dim is not None:
        pca = PCA(opt.pca_dim)
        gallery_features = pca.fit_transform(gallery_features)
        query_feature = pca.transform(query_feature)

    gallery_features /= np.linalg.norm(gallery_features, axis=-1, keepdims=True)
    query_feature /= np.linalg.norm(query_feature, axis=-1, keepdims=True)

    identifiers = np.asarray(ids)

    if opt.rerank:
        k2 = k
        k1 = 3 * k2

        gallery_features = torch.FloatTensor(gallery_features)
        query_feature = torch.FloatTensor(query_feature)
        query_feature = query_feature.cuda()
        gallery_features = gallery_features.cuda()

        # Feature vectors need to be normed
        order = np.squeeze(gnn_reranking(query_feature, gallery_features, k1, k2))[:k2]
    else:
        order = np.argsort(np.squeeze(cosine_distances(gallery_features, query_feature)))[:k]

    identifiers = identifiers[order]

    anchor = cv2.imread(identifiers[0].item())
    print(f'Rank 0: {identifiers[0].item()}')
    window = f'Compare'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    for rank, filename in enumerate(identifiers[1:], 1):
        print(f'Rank {rank}: {filename}')
        image = cv2.imread(filename.item())
        cv2.imshow(window, np.concatenate((anchor, image), axis=1))
        cv2.waitKey()
    cv2.destroyAllWindows()

    for filename in identifiers[1:]:
        print(identifiers[0].split('/')[-1], filename.split('/')[-1])
