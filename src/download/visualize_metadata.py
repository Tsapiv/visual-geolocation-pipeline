from argparse import ArgumentParser

import gmplot
import matplotlib

from camera import CameraMetadata
from dataset import Dataset

matplotlib.use('TkAgg')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Metadate path')

    opt = parser.parse_args()
    dataset = Dataset(opt.input)

    md = list(map(lambda x: CameraMetadata.from_kwargs(**x), dataset.metadata(dataset.entries)))

    coords = list(set(list(map(lambda x: (x.lat, x.lng), md))))

    apikey = ''  # (your API key here)
    gmap = gmplot.GoogleMapPlotter(49.8443931, 24.0254815, 17, apikey=apikey)

    gmap.scatter(
        *zip(*coords), color='blue', marker=False, size=2,
    )

    gmap.draw('map.html')
