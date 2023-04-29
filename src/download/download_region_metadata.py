import json
import os
from argparse import ArgumentParser
from uuid import uuid4

import requests
import yaml
from tqdm import tqdm

from region_selection import get_coords_around_point
from utils import sign_url

META_REQUEST = 'https://maps.googleapis.com/maps/api/streetview/metadata?location={location}&key={key}'
DATA_REQUEST = 'https://maps.googleapis.com/maps/api/streetview?location={location}&size={size}&heading={heading}&key={key}&source=outdoor'

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', nargs='+', type=float, required=True, help='Define point')
    parser.add_argument('-j', nargs='+', type=float, default=None, help='Jitter range in meters')
    parser.add_argument('-r', type=float, default=1000, help='Radius in meters')
    parser.add_argument('-s', default='data', type=str, help='Option for save')
    parser.add_argument('-d', type=float, default=10, help='Distance between neighboring nodes in meters')

    opt = parser.parse_args()

    cred = yaml.safe_load(open('credentials/google.yaml'))
    root = os.path.join(opt.s, "D" + "@".join(map(str, opt.p)))
    os.makedirs(root, exist_ok=True)
    check = set()
    jitter = tuple(opt.j) if opt.j is not None else None
    for coord in tqdm(get_coords_around_point(tuple(opt.p), opt.r, opt.d, jitter, False)):
        try:
            params = dict(location=",".join(map(str, coord)), key=cred['api-key'])
            response = requests.get(sign_url(META_REQUEST.format(**params), cred['secret']), stream=True, timeout=10).json()
            status = response['status']

            if status.lower() != 'ok':
                print(f'Status: {status}')
                continue

            location = response['location']
            pano = response['pano_id']

            if pano in check:
                continue

            check.add(pano)

            os.makedirs(os.path.join(root, pano))
            json.dump({**location, 'pano': pano}, open(os.path.join(root, pano, 'metadata.json'), 'w'), indent=4)

        except Exception as e:
            print(e)
