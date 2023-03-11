import json
import os
import shutil
from argparse import ArgumentParser
from uuid import uuid4

import numpy as np
import tqdm


def convert_elevation(inp):
    output = {}
    for entry in inp:
        output[(entry['location']['lat'], entry['location']['lng'])] = entry['elevation']
    return output


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', required=True, type=str, help='Path to directory with photos')
    parser.add_argument('--features', required=True, type=str, help='Path to directory with photos')
    parser.add_argument('--elevation', required=True, help='Comma separated coordinate lat@lng')
    parser.add_argument('--output', required=True, type=str, help='Path to directory with photos')

    opt = parser.parse_args()

    os.makedirs(opt.output, exist_ok=True)

    image_paths = list(os.scandir(opt.input))
    ids = list(map(lambda x: os.path.basename(x), json.load(open(f'{opt.features}-id.json'))))
    features = np.squeeze(np.load(f'{opt.features}-features.npy'))

    assert len(image_paths) == len(ids) == len(features)

    elevation = convert_elevation(json.load(open(opt.elevation)))

    for entry in tqdm.tqdm(image_paths):

        coords, azn = entry.name.rstrip('.jpg').split('_')
        lat, lng = coords.split('@')
        lat = float(lat)
        lng = float(lng)
        azn = float(azn)
        alt = elevation[(lat, lng)]
        metadata = dict(lat=lat, lng=lng, alt=alt, azn=azn, w=640, h=640, fov=90)
        idx = ids.index(entry.name)

        descriptor = features[idx]
        save_dir = os.path.join(opt.output, str(uuid4()))
        os.makedirs(save_dir)

        json.dump(metadata, open(os.path.join(save_dir, 'metadata.json'), 'w'), indent=4)
        descriptor_path = os.path.join(save_dir, f'{os.path.basename(opt.features)}_descriptor.npy')
        np.save(descriptor_path, descriptor)
        shutil.copyfile(entry.path, os.path.join(save_dir, 'image.jpg'))

