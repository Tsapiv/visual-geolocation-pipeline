import json
import os
from argparse import ArgumentParser

import requests
import yaml
from tqdm import tqdm

from dataset import Dataset

DATA_REQUEST = 'https://maps.googleapis.com/maps/api/elevation/json?locations={locations}&key={key}'


def split(list_, chunk_size):
    for i in range(0, len(list_), chunk_size):
        yield list_[i:i + chunk_size]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='Path to dir with images')

    opt = parser.parse_args()

    cred = yaml.safe_load(open('credentials/google.yaml'))

    dataset = Dataset(opt.dir)

    coords = [f"{m['lat']},{m['lng']}" for m in dataset.metadata(dataset.entries, cache=True)]

    metadata = []
    for coord in tqdm(list(split(coords, 300))):
        try:
            params = dict(locations="|".join(coord), key=cred['api-key'])
            response = requests.get(DATA_REQUEST.format(**params)).json()
            metadata.extend(response['results'])
        except Exception as e:
            print(e)

    for i, entry in enumerate(dataset.entries):
        m = dataset.metadata(entry)
        m.update(alt=metadata[i]['elevation'])
        json.dump(m, open(os.path.join(dataset.root, entry, 'metadata.json'), 'w'), indent=4)
