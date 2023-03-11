import json
import os
from argparse import ArgumentParser

import requests
import yaml

from utils import sign_url

ROUTE_REQUEST = 'https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&key={key}'

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src', nargs='+', type=float, required=True, help='Origin point')
    parser.add_argument('--dst', nargs='+', type=float, required=True, help='Destination point')
    parser.add_argument('-s', default='data', type=str, help='Option for save')

    opt = parser.parse_args()

    cred = yaml.safe_load(open('credentials/google.yaml'))
    src = ",".join(map(str, opt.src))
    dst = ",".join(map(str, opt.dst))

    os.makedirs(opt.s, exist_ok=True)

    try:
        params = dict(origin=src, destination=dst, key=cred['api-key'])
        route = requests.get(ROUTE_REQUEST.format(**params), stream=True).json()
        json.dump(route, open(os.path.join(opt.s, f'route-{src}-{dst}.json'), 'w'), indent=4)

    except Exception as e:
        print(e)
