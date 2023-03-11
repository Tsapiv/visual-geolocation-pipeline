import json
import os

IMAGE_LIST = 'assets/scene.txt'
PHOTO_DIR = 'data/49.8443931@24.0254815/photo'
ROOT_DIR = 'assets/franko/'

def convert_elevation(inp):
    output = {}
    for entry in inp:
        output[(entry['location']['lat'], entry['location']['lng'])] = entry['elevation']
    return output

if __name__ == '__main__':
    elevation = convert_elevation(json.load(open('data/49.8443931@24.0254815/elevation.json')))
    exif = {}
    for image_name in open('assets/scene.txt'):
        image_name = image_name.strip()
        os.system(f'cp {os.path.join(PHOTO_DIR, image_name)} {os.path.join(ROOT_DIR, "images", image_name)}')

        coords = image_name.split('_')[0]
        lat, lng = list(map(float, coords.split('@')))
        exif[image_name] = {'gps': {
            "latitude": lat,
            "longitude": lng,
            "altitude": elevation[(lat, lng)],
            "dop": 5.0
        }}
    json.dump(exif, open(os.path.join(ROOT_DIR, 'exif_overrides.json'), 'w'), indent=4)

