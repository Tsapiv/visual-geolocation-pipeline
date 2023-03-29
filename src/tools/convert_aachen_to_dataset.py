import json
import math
import os.path
import shutil
from typing import Dict, Any
from uuid import uuid4

import cv2
import numpy as np
from tqdm import tqdm

INPUT = '/home/tsapiv/Downloads/aachen_v1/'
OUTPUT = 'datasets/aachen_v1_nighttime_test_resized'
RESIZE = 800


def from_quat_to_mat(cam_rot):
    angle = 2 * math.acos(cam_rot[0])
    x = cam_rot[1] / math.sqrt(1 - cam_rot[0] ** 2)
    y = cam_rot[2] / math.sqrt(1 - cam_rot[0] ** 2)
    z = cam_rot[3] / math.sqrt(1 - cam_rot[0] ** 2)

    cam_rot = [x * angle, y * angle, z * angle]

    cam_rot = np.asarray(cam_rot)
    cam_rot, _ = cv2.Rodrigues(cam_rot)
    return cam_rot


if __name__ == '__main__':

    os.makedirs(OUTPUT, exist_ok=True)
    metadata: Dict[str, Dict[str, Any]] = {}

    intrinsic_path = os.path.join(INPUT, 'night_time_queries_with_intrinsics.txt')
    with open(intrinsic_path) as f:
        for line in f.readlines():
            filename, _, *parameters = line.strip().split(' ')
            w, h, focal, cx, cy, r = list(map(float, parameters))
            if RESIZE is not None:
                scale = RESIZE / max(w, h)

                w = int(round(w * scale))
                h = int(round(h * scale))
                focal *= scale
                cx *= scale
                cy *= scale

            K = [[focal, 0, cx],
                 [0, focal, cy],
                 [0, 0, 1]]
            metadata[filename] = dict(K=K, h=h, w=w)

    extrinsic_path = os.path.join(INPUT, 'aachen_cvpr2018_db.nvm')
    with open(extrinsic_path) as f:
        next(f)
        next(f)
        n = int(f.readline().strip())
        for _ in range(n):
            line = f.readline().strip()
            filename, *parameters = line.split(' ')
            _, rw, rx, ry, rz, cx, cy, cz, _, _ = list(map(float, parameters))
            C = np.asarray([cx, cy, cz])
            R = from_quat_to_mat(np.asarray([rw, rx, ry, rz]))
            E = np.eye(4)
            T = -R @ C
            E[:3, :3] = R
            E[:3, -1] = T

            metadata[filename].update(E=E.tolist())
    for image_path, meta in tqdm(metadata.items()):
        if not image_path.endswith('.jpg'):
            print('Wrong file format')
            continue
        uuid = str(uuid4())
        os.makedirs(os.path.join(OUTPUT, uuid))
        if RESIZE is None:
            shutil.copy(os.path.join(INPUT, image_path), os.path.join(OUTPUT, uuid, 'image.jpg'))
        else:
            image = cv2.imread(os.path.join(INPUT, image_path))
            resized = cv2.resize(image, (meta['w'], meta['h']), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(OUTPUT, uuid, 'image.jpg'), resized)

        json.dump(meta, open(os.path.join(OUTPUT, uuid, 'metadata.json'), 'w'), indent=4)
