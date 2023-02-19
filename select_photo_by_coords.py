import os
from argparse import ArgumentParser

import cv2
import numpy as np
from pyproj import Geod

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', required=True, type=str, help='Path to directory with photos')
    parser.add_argument('-c', required=True, help='Comma separated coordinate lat@lng')
    parser.add_argument('-k', required=False, type=int, default=3)
    parser.add_argument('-v', action='store_true', help='Option for visualization')
    parser.add_argument('-f', action='store_true', help='Option for filtering')

    opt = parser.parse_args()

    geod = Geod(ellps='WGS84')

    image_filenames = list(os.scandir(opt.input))

    image_coords = list(map(lambda x: list(map(float, x.name.split('_')[0].split('@'))), image_filenames))

    query_lat, query_lng = list(map(float, opt.c.split('@')))

    _, _, distances = geod.inv([query_lng] * len(image_coords), [query_lat] * len(image_coords), *(list(zip(*image_coords))[::-1]))
    distances = np.asarray(distances)
    order = np.argsort(distances)[:opt.k]

    window = f'Closes to {opt.c}'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    filtered = []
    for idx in order:

        if opt.v or opt.f:
            cv2.imshow(window, cv2.imread(image_filenames[idx].path))
            cv2.waitKey()
            if opt.f:
                keep = input('Keep image [Y/n]: ')
                if keep.lower() == 'y' or keep == '':
                    filtered.append(image_filenames[idx].path)

        if not opt.f:
            print(image_filenames[idx].path)

    print('\n'.join(filtered))

    cv2.destroyAllWindows()

