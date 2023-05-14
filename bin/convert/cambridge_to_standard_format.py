import json
import os.path
from argparse import ArgumentParser

import cv2
import numpy as np
import tqdm
from scipy.spatial.transform import Rotation

WIDTH = 852
HEIGHT = 480
FOCAL_LENGTH = 744.375

K = [[FOCAL_LENGTH, 0, WIDTH / 2],
     [0, FOCAL_LENGTH, HEIGHT / 2],
     [0, 0, 1]]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path file with camera data')
    parser.add_argument('--dataset', type=str, required=True, help='Path to where create dataset')

    opt = parser.parse_args()

    os.makedirs(opt.dataset, exist_ok=True)
    with open(opt.input) as f:
        lines = f.readlines()[3:]
        base_dir = os.path.dirname(opt.input)
        for line in tqdm.tqdm(lines):
            image_path, *camera_pose = line.strip().split()
            camera_pose = list(map(float, camera_pose))

            C = camera_pose[:3]
            R = Rotation.from_quat(camera_pose[4:] + [camera_pose[3]]).as_matrix()
            E = np.eye(4)
            T = -R @ C
            E[:3, :3] = R
            E[:3, -1] = T

            image = cv2.imread(os.path.join(base_dir, image_path))

            resized = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)

            metadata = dict(h=HEIGHT, w=WIDTH, K=K, E=E.tolist())
            uuid = image_path.replace('/', '_')
            os.makedirs(os.path.join(opt.dataset, uuid))
            cv2.imwrite(os.path.join(opt.dataset, uuid, 'image.jpg'), resized)
            json.dump(metadata, open(os.path.join(opt.dataset, uuid, 'metadata.json'), 'w'), indent=4)
