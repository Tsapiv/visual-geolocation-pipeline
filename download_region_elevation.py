import json
import os
from argparse import ArgumentParser

import requests
import yaml
from tqdm import tqdm

from utils import sign_url

DATA_REQUEST = 'https://maps.googleapis.com/maps/api/elevation/json?locations={locations}&key={key}'


def split(list_a, chunk_size):
    for i in range(0, len(list_a), chunk_size):
        yield list_a[i:i + chunk_size]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='Path to dir with images')
    parser.add_argument('--output', type=str, default='./', help='Path to save json')

    opt = parser.parse_args()

    cred = yaml.safe_load(open('credentials/google.yaml'))

    coords = list(set(
        map(lambda y: y.split('_')[0].replace('@', ','), map(lambda x: x.name, os.scandir(opt.dir)))))
    meta = []
    for coord in tqdm(list(split(coords, 200))):
        try:
            params = dict(locations="|".join(coord), key=cred['api-key'])
            response = requests.get(DATA_REQUEST.format(**params))
            metadata = response.json()
            meta.extend(metadata['results'])
        except Exception as e:
            print(e)
    json.dump(meta, open(os.path.join(opt.output, 'elevation.json'), 'w'), indent=4)
