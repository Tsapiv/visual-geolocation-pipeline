import json
import os.path

import tqdm

from dataset import Dataset

INPUT = '/home/tsapiv/Documents/diploma/ShopFacade/reconstruction.nvm'
OUTPUT = 'datasets/ShopFacadeTest'

if __name__ == '__main__':

    dataset = Dataset(OUTPUT)

    entries = set(dataset.entries)

    with open(INPUT) as f:
        lines = f.readlines()
        n = int(lines[2])
        lines = lines[3: n + 3]
        base_dir = os.path.dirname(INPUT)
        for line in tqdm.tqdm(lines):
            info = line.strip().split()
            image_path = info[0]
            r = float(info[-2])

            uuid = image_path.replace('/', '_').replace('.jpg', '.png')
            if uuid not in entries:
                continue

            metadata = dataset.metadata(uuid)

            metadata.update(distortion_coefficients=[r, 0, 0, 0, 0])

            json.dump(metadata, open(os.path.join(OUTPUT, uuid, 'metadata.json'), 'w'), indent=4)
