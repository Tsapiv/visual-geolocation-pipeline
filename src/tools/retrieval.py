import os
from argparse import ArgumentParser

import cv2
import numpy as np

from dataset import Dataset
from ranking.index import Index

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str, help='Path to root directory of the database')
    parser.add_argument('-d', '--descriptor', required=True, type=str, help='Name of descriptor type')
    parser.add_argument('-k', required=False, type=int, default=3)
    parser.add_argument('--rerank', action='store_true', help='Option for reranking')
    parser.add_argument('-q', '--query', default='4000', type=str, help='Name/number of query picture')

    opt = parser.parse_args()

    dataset = Dataset(opt.dataset, opt.descriptor)
    index = Index(dataset)

    k = opt.k + 1
    try:
        n = int(opt.query)
        query_descriptor = dataset.descriptor(dataset.entries[n]).reshape(1, -1)
    except ValueError:
        if os.path.isdir(opt.query):
            query_descriptor = np.load(os.path.join(opt.query, f'{opt.descriptor}_descriptor.npy')).reshape(1, -1)
        else:
            query_descriptor = np.load(opt.query).reshape(1, -1)
    topk = index.topk(query_descriptor, k, opt.rerank)
    print(topk)
    images = list(dataset.image(topk))

    anchor = images[0]
    window = f'Compare'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    for rank, image in enumerate(images[1:], 1):
        cv2.imshow(window, np.concatenate((anchor, image), axis=1))
        cv2.waitKey()
    cv2.destroyAllWindows()
