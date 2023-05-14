from argparse import ArgumentParser

import gmplot
import matplotlib
import numpy as np
import yaml

from pysight.camera import CameraMetadata
from pysight.dataset import Dataset

matplotlib.use('TkAgg')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset path')
    parser.add_argument('--credentials', type=str, default=None, help='Path to credentials file')
    parser.add_argument('--zoom', type=int, default=17, help='Starting zoom of the map')
    parser.add_argument('--output', type=str, default='map.html', help='HTML output path')

    opt = parser.parse_args()

    dataset = Dataset(opt.dataset)

    md = list(map(lambda x: CameraMetadata.from_kwargs(**x), dataset.metadata(dataset.entries)))

    coords = np.asarray(list(set(list(map(lambda x: (x.lat, x.lng), md)))))

    if opt.credentials is None:
        apikey = ''
    else:
        apikey = yaml.safe_load(open(opt.credentials))

    gmap = gmplot.GoogleMapPlotter(*coords.mean(axis=0), opt.zoom, apikey=apikey)

    gmap.scatter(*zip(*coords), color='blue', marker=False, size=2)

    gmap.draw(opt.output)
