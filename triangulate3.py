import json
import os

import cv2
import numpy as np
from pyproj import Geod, Proj
from scipy.spatial.transform import Rotation as Rot

PATH = 'matches2'

proj = Proj(proj='utm', zone=35, ellps='WGS84')

geod = Geod(ellps='WGS84')

w, h = 640, 640
fov = 90

def convert_elevation(inp):
    output = {}
    for entry in inp:
        output[(entry['location']['lat'], entry['location']['lng'])] = entry['elevation']
    return output

elevation = convert_elevation(json.load(open('data/49.8443931@24.0254815/elevation.json')))

K = np.array([[w / np.tan(np.deg2rad(fov / 2)) / 2, 0, w / 2],
              [0, h / np.tan(np.deg2rad(fov / 2)) / 2, h / 2],
              [0, 0, 1]])


def get_projection_matrix(K, R, T):
    return np.concatenate((np.dot(K,R),np.dot(K,T).reshape(-1, 1)), axis = 1)


def parse_relative_motion(name: str):
    coords1, heading1, coords2, heading2, _ = name.split('_')

    heading1_deg = float(heading1)
    lat1, lng1 = list(map(float, coords1.split('@')))

    heading2_deg = float(heading2)
    lat2, lng2 = list(map(float, coords2.split('@')))

    # azimuths_deg, _, dist = geod.inv(lng1, lat1, lng2, lat2)

    # T12 = np.array([np.sin(np.deg2rad(azimuths_deg)) * dist, 0, np.cos(np.deg2rad(azimuths_deg)) * dist])

    # _, _, dx = geod.inv(lng1, lat1, lng1, lat2)
    # _, _, dy = geod.inv(lng1, lat1, lng2, lat1)
    # T12 = np.array([dx, 0, dy])

    Y1 = elevation[(lat1, lng1)]
    Y2 = elevation[(lat2, lng2)]

    X1, Z1 = proj(lng1, lat1)
    X2, Z2 = proj(lng2, lat2)
    # X1, Z1 = proj(lat1, lng1)
    # X2, Z2 = proj(lat2, lng2)
    T12 = np.array([X2 - X1, Y2 - Y1, Z2 - Z1])[::-1]
    R12 = Rot.from_rotvec(np.array([0, -np.deg2rad(heading2_deg - heading1_deg), 0])).as_matrix()

    return T12, R12

if __name__ == '__main__':

    archives = []
    extrinsics = []
    for dir_entry in os.scandir(PATH):
        if not dir_entry.name.endswith('npz'):
            continue
        archives.append(np.load(dir_entry.path))
        extrinsics.append(parse_relative_motion(dir_entry.name))


    for (T, R), arch in zip(extrinsics, archives):
        if np.allclose(T, [0, 0, 0]):
            continue
        kpts1, kpts2 = arch['keypoints0'], arch['keypoints1']

        mask = arch['match_confidence'] > 0.5
        matches = arch['matches'][mask]

        mkpts1 = kpts1[mask]
        mkpts2 = kpts2[matches]

        F_pred, inliers_mask = cv2.findFundamentalMat(
            mkpts1, mkpts2,
            cv2.USAC_MAGSAC,
            ransacReprojThreshold=0.5,
            confidence=0.99999,
            maxIters=10000)

        E_pred = K.T @ F_pred @ K
        n, R_pred, T_pred, _ = cv2.recoverPose(E_pred, mkpts1, mkpts2, np.eye(3), 1e9, mask=inliers_mask)
        T_pred = T_pred[:, 0]

        print(T_pred)
        print(R_pred)
        print(Rot.from_matrix(R_pred).as_rotvec(degrees=True))
        print(np.linalg.norm(T_pred - T))


