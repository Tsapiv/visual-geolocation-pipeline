import json
import os
from copy import deepcopy

import requests
import yaml
from tqdm import tqdm

from ..dataset import Dataset
from ..utils import sign_url

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 640

DATA_REQUEST = 'https://maps.googleapis.com/maps/api/streetview?pano={pano}&size=640x640&fov={fov}&heading={heading}&key={key}&source=outdoor'


def download_panoramas(dataset_path: str, credential_path: str, fov: int, n_directions: int):
    cred = yaml.safe_load(open(credential_path))
    dataset = Dataset(dataset_path)
    params = dict(key=cred['api-key'], fov=fov)
    for entry in tqdm(dataset.entries):
        try:
            metadata_base = dataset.metadata(entry)
            params.update(metadata_base)
            for heading in range(0, 360, 360 // n_directions):

                entry_path = os.path.join(dataset.root, entry)

                if os.path.exists(os.path.join(entry_path, 'image.jpg')):
                    if metadata_base.get('azn', None) == heading:
                        continue
                    entry_path = os.path.join(dataset.root, f'{entry}_{heading}')
                    os.makedirs(entry_path)

                photo_path = os.path.join(entry_path, 'image.jpg')

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

                metadata = deepcopy(metadata_base)
                metadata.update(dict(fov=fov, w=DEFAULT_WIDTH, h=DEFAULT_HEIGHT, azn=heading))
                metadata_path = os.path.join(entry_path, 'metadata.json')
                json.dump(metadata, open(metadata_path, 'w'), indent=4)

        except Exception as e:
            print(e)
# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('--metadata', type=str, required=True, help='Path to metadata file')
#     parser.add_argument('--fov', type=int, default=90, help='Field of view')
#     parser.add_argument('--headings', type=int, default=8, help='Number of headings')
#
#     opt = parser.parse_args()
#
#     cred = yaml.safe_load(open('credentials/google.yaml'))
#     dataset = Dataset(opt.metadata)
#     params = dict(key=cred['api-key'], fov=opt.fov)
#     for entry in tqdm(dataset.entries):
#         try:
#             metadata_base = dataset.metadata(entry)
#             params.update(metadata_base)
#             for heading in range(0, 360, 360 // opt.headings):
#
#                 entry_path = os.path.join(dataset.root, entry)
#
#                 if os.path.exists(os.path.join(entry_path, 'image.jpg')):
#                     if metadata_base.get('azn', None) == heading:
#                         continue
#                     entry_path = os.path.join(dataset.root, f'{entry}_{heading}')
#                     os.makedirs(entry_path)
#
#                 photo_path = os.path.join(entry_path, 'image.jpg')
#
#                 params['heading'] = heading
#                 request = DATA_REQUEST.format(**params)
#                 request = sign_url(request, cred['secret'])
#                 response = requests.get(request)
#                 if not response.ok:
#                     print(response.status_code)
#                     continue
#                 with open(photo_path, 'wb') as file:
#                     file.write(response.content)
#                 response.close()
#
#                 metadata = deepcopy(metadata_base)
#                 metadata.update(dict(fov=opt.fov, w=640, h=640, azn=heading))
#                 metadata_path = os.path.join(entry_path, 'metadata.json')
#                 json.dump(metadata, open(metadata_path, 'w'), indent=4)
#
#         except Exception as e:
#             print(e)
