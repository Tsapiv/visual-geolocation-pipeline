import json
import os.path
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation
import cv2


INPUT = '/home/tsapiv/Documents/diploma/KingsCollege/dataset_test.txt'
OUTPUT = 'datasets/KingsCollegeTest'

fl = 744.375
xs = 0.0
ys = 0.0

w = 852
h = 480

K = [[fl, 0, xs],
     [0, fl, ys],
     [0, 0, 1]]

if __name__ == '__main__':

    os.makedirs(OUTPUT, exist_ok=True)
    with open(INPUT) as f:
        lines = f.readlines()[3:]
        base_dir = os.path.dirname(INPUT)
        for line in lines:
            image_path, *camera_pose = line.strip().split()
            camera_pose = list(map(float, camera_pose))

            T = camera_pose[:3]
            R = Rotation.from_quat(camera_pose[3:]).as_matrix()
            E = np.eye(4)
            E[:3, :3] = R
            E[:3, -1] = T

            image = cv2.imread(os.path.join(base_dir, image_path))

            resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)

            metadata = dict(h=h, w=w, K=K, E=E.tolist())
            uuid = str(uuid4())
            os.makedirs(os.path.join(OUTPUT, uuid))
            cv2.imwrite(os.path.join(OUTPUT, uuid, 'image.jpg'), resized)
            json.dump(metadata, open(os.path.join(OUTPUT, uuid, 'metadata.json'), 'w'), indent=4)