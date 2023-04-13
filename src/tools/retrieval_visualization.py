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

    shapes = np.asarray(list(map(lambda x: x.shape[:2], images)))
    max_h, max_w = np.max(shapes, axis=0)
    anchor = images[0]
    anchor_padded = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    anchor_padded[:anchor.shape[0], :anchor.shape[1], :] = anchor
    window = f'Compare'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    for rank, image in enumerate(images[1:]):
        image_padded = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        image_padded[:image.shape[0], :image.shape[1], :] = image
        cv2.imshow(window, np.concatenate((anchor_padded, image_padded), axis=1))
        cv2.waitKey()
    cv2.destroyAllWindows()
