import os
import pathlib
from argparse import ArgumentParser

import requests
import yaml
from tqdm import tqdm

from utils import sign_url

DATA_REQUEST = 'https://maps.googleapis.com/maps/api/streetview?pano={pano}&size=640x640&fov={fov}&heading={heading}&key={key}&source=outdoor'

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--metadata', type=str, required=True, help='Path to metadata file')
    parser.add_argument('--fov', type=int, default=90, help='Field of view')
    parser.add_argument('--headings', type=int, default=8, help='Number of headings')

    opt = parser.parse_args()

    cred = yaml.safe_load(open('credentials/google.yaml'))
    metadata = yaml.safe_load(open(opt.metadata))
    root = str(pathlib.Path(opt.metadata).parent)
    os.makedirs(os.path.join(root, 'photo'), exist_ok=True)
    params = dict(key=cred['api-key'], fov=opt.fov)
    for pano in tqdm(metadata):
        try:
            params['pano'] = pano['pano_id']
            for heading in range(0, 360, 360 // opt.headings):
                lat, lng = pano['location']['lat'], pano['location']['lng']
                photo_path = os.path.join(root, 'photo', f'{lat}@{lng}_{heading}.jpg')
                if os.path.exists(photo_path):
                    continue

                params['heading'] = heading
                request = DATA_REQUEST.format(**params)
                request = sign_url(request, cred['secret'])
                response = requests.get(request)
                if not response.ok:
                    print(response.status_code)
                    continue
                with open(photo_path, 'wb') as file:
                    file.write(response.content)
                response.close()
        except Exception as e:
            print(e)
