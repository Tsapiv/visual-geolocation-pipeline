import json
import os
from argparse import ArgumentParser

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
    parser.add_argument('-r', type=float, default=1000, help='Radius in meters')
    parser.add_argument('-s', default='data', type=str, help='Option for save')
    parser.add_argument('--positions', type=int, default=1, help='Number of directions')

    args = parser.parse_args()

    cred = yaml.safe_load(open('credentials/google.yaml'))
    root = os.path.join(args.s, ",".join(map(str, args.p)))
    os.makedirs(os.path.join(root, 'photo'))
    meta = []
    for coord in tqdm(get_coords_around_point(tuple(args.p), args.r, args.v)):
        try:
            params = dict(location=",".join(map(str, coord)), size='640x640', heading=0, key=cred['api-key'])
            metadata = requests.get(sign_url(META_REQUEST.format(**params), cred['secret']), stream=True).json()
            params['location'] = f"{metadata['location']['lat']},{metadata['location']['lng']}"
            for heading in range(0, 360, 360 // args.positions):
                params['heading'] = heading
                request = DATA_REQUEST.format(**params)
                request = sign_url(request, cred['secret'])
                response = requests.get(request)
                if not response.ok:
                    print(response.status_code)
                with open(os.path.join(root, 'photo', f'{params["location"]}_{heading}.jpg'), 'wb') as file:
                    file.write(response.content)
                response.close()
            meta.append(metadata)
        except Exception as e:
            print(e)
    json.dump(meta, open(os.path.join(root, 'metadata.json'), 'w'), indent=4)
