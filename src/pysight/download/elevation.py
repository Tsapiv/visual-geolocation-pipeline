import json
import os

import requests
import yaml
from tqdm import tqdm

from ..dataset import Dataset

DATA_REQUEST = 'https://maps.googleapis.com/maps/api/elevation/json?locations={locations}&key={key}'


def _split(collection, chunk_size):
    for i in range(0, len(collection), chunk_size):
        yield collection[i:i + chunk_size]


def download_elevation(dataset_path: str, credential_path: str, chunk_size: int = 300):
    cred = yaml.safe_load(open(credential_path))

    dataset = Dataset(dataset_path)

    coords = [f"{m['lat']},{m['lng']}" for m in dataset.metadata(dataset.entries, cache=True)]

    metadata = []
    for coord in tqdm(list(_split(coords, chunk_size))):
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
