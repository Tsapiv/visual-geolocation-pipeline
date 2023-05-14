import json
import os.path
from argparse import ArgumentParser
from typing import Dict, Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--intrinsics', type=str, required=True, help='Path file with camera intrinsic data')
    parser.add_argument('--extrinsics', type=str, default=None, help='Path file with camera extrinsic data')
    parser.add_argument('--dataset', type=str, required=True, help='Path to where create dataset')
    parser.add_argument('--resize', type=int, default=None, help='Resize larger side to that value')

    opt = parser.parse_args()

    os.makedirs(opt.dataset, exist_ok=True)
    metadata: Dict[str, Dict[str, Any]] = {}

    with open(opt.intrinsics) as f:
        for line in f.readlines():
            filename, _, *parameters = line.strip().split(' ')
            w, h, focal, cx, cy, r = list(map(float, parameters))
            if opt.resize is not None and opt.resize < max(w, h):
                scale = opt.resize / max(w, h)

                w = int(round(w * scale))
                h = int(round(h * scale))
                focal *= scale
                cx *= scale
                cy *= scale

            K = [[focal, 0, cx],
                 [0, focal, cy],
                 [0, 0, 1]]
            metadata[filename] = dict(K=K, h=int(h), w=int(w), distortion_coefficients=[r, 0, 0, 0, 0])

    if opt.extrinsics is not None:
        with open(opt.extrinsics) as f:
            next(f)
            next(f)
            n = int(f.readline().strip())
            for _ in range(n):
                line = f.readline().strip()
                filename, *parameters = line.split(' ')
                _, rw, rx, ry, rz, cx, cy, cz, _, _ = list(map(float, parameters))
                C = np.asarray([cx, cy, cz])
                R = Rotation.from_quat([rx, ry, rz, rw]).as_matrix()
                E = np.eye(4)
                T = -R @ C
                E[:3, :3] = R
                E[:3, -1] = T

                metadata[filename].update(E=E.tolist())

    for image_path, meta in tqdm(metadata.items()):
        abs_image_path = os.path.join(os.path.dirname(opt.intrinsics), image_path)
        if not os.path.exists(abs_image_path):
            continue

        uuid = image_path.rstrip('.jpg').rstrip('.png').split('/')[-1]
        os.makedirs(os.path.join(opt.dataset, uuid))

        image = cv2.imread(abs_image_path)
        resized = cv2.resize(image, (meta['w'], meta['h']), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(opt.dataset, uuid, 'image.jpg'), resized)

        json.dump(meta, open(os.path.join(opt.dataset, uuid, 'metadata.json'), 'w'), indent=4)
