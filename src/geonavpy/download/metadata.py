import json
import os
from typing import Iterable, Tuple

import requests
import yaml
from tqdm import tqdm

from ..utils import sign_url

META_REQUEST = 'https://maps.googleapis.com/maps/api/streetview/metadata?location={location}&key={key}'
DATA_REQUEST = 'https://maps.googleapis.com/maps/api/streetview?location={location}&size={size}&heading={heading}&key={key}&source=outdoor'


def download_metadata(dataset_path: str, credential_path: str, coordinates: Iterable[Tuple]):
    cred = yaml.safe_load(open(credential_path))
    os.makedirs(dataset_path, exist_ok=True)
    check = set()
    for coord in tqdm(coordinates):
        try:
            params = dict(location=",".join(map(str, coord)), key=cred['api-key'])
            response = requests.get(sign_url(META_REQUEST.format(**params), cred['secret']), stream=True,
                                    timeout=10).json()
            status = response['status']

            if status.lower() != 'ok':
                print(f'Status: {status}')
                continue

            location = response['location']
            pano = response['pano_id']

            if pano in check:
                continue

            check.add(pano)

            os.makedirs(os.path.join(dataset_path, pano))
            json.dump({**location, 'pano': pano}, open(os.path.join(dataset_path, pano, 'metadata.json'), 'w'),
                      indent=4)

        except Exception as e:
            print(e)
