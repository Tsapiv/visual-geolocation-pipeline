import json
import math
import os.path
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation
import cv2


INPUT = '/home/tsapiv/Documents/diploma/KingsCollege/dataset_train.txt'
OUTPUT = 'datasets/KingsCollegeTrain'

w = 852
h = 480

fl = 744.375
xs = w / 2
ys = h / 2


K = [[fl, 0, xs],
     [0, fl, ys],
     [0, 0, 1]]

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
    with open(INPUT) as f:
        lines = f.readlines()[3:]
        base_dir = os.path.dirname(INPUT)
        for line in lines:
            image_path, *camera_pose = line.strip().split()
            camera_pose = list(map(float, camera_pose))

            C = camera_pose[:3]
            R = from_quat_to_mat(camera_pose[3:])
            E = np.eye(4)
            T = -R @ C
            E[:3, :3] = R
            E[:3, -1] = T

            image = cv2.imread(os.path.join(base_dir, image_path))

            resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)

            metadata = dict(h=h, w=w, K=K, E=E.tolist())
            uuid = str(uuid4())
            os.makedirs(os.path.join(OUTPUT, uuid))
            cv2.imwrite(os.path.join(OUTPUT, uuid, 'image.jpg'), resized)
            json.dump(metadata, open(os.path.join(OUTPUT, uuid, 'metadata.json'), 'w'), indent=4)