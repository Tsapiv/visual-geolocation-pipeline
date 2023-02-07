import os

import numpy as np
from itertools import combinations
from pyproj import Geod
from scipy.spatial.transform import Rotation as R
from position_refinement.LinearTriangulation import LinearTriangulation
from position_refinement.PnPRansac import PnPRANSAC
from position_refinement.NonLinearPnP import NonLinearPnP
from position_refinement.NonLinearTriangulation import NonLinearTriangulation

PATH = 'dump_match_pairs'

geod = Geod(ellps='WGS84')

w, h = 640, 640
fov = 90

K = np.array([[w / np.tan(np.deg2rad(fov / 2)), 0, w / 2],
              [0, h / np.tan(np.deg2rad(fov / 2)), h / 2],
              [0, 0, 1]])


def parse_relative_motion(name: str):
    coords1, heading1, coords2, heading2, _ = name.split('_')

    heading1_deg = float(heading1)
    lat1, lng1 = list(map(float, coords1.split('@')))

    heading2_deg = float(heading2)
    lat2, lng2 = list(map(float, coords2.split('@')))

    azimuths_deg, _, dist = geod.inv(lng1, lat1, lng2, lat2)

    T12 = np.array([np.sin(np.deg2rad(azimuths_deg)) * dist, 0, np.cos(np.deg2rad(azimuths_deg)) * dist])
    R12 = R.from_rotvec(np.array([0, np.deg2rad(heading2_deg - heading1_deg), 0])).as_matrix()

    return T12, R12


if __name__ == '__main__':
    print(K)
    archives = []
    extrinsics = []
    for dir_entry in os.scandir(PATH):
        if not dir_entry.name.endswith('npz'):
            continue
        archives.append(np.load(dir_entry.path))
        extrinsics.append(parse_relative_motion(dir_entry.name))

    anchor_kpts = archives[0]['keypoints0']

    kpts = []
    matches = []
    common_matches = np.ones_like(archives[0]['matches'])
    for arch in archives:
        kpts.append(arch['keypoints1'])
        matches.append(np.where(arch['matches'] == -1, 0, arch['matches']))

    pts3d = []
    pts2d = []

    for (match1, pts1, extrinsic1), (match2, pts2, extrinsic2) in combinations(zip(matches, kpts, extrinsics), 2):
        idxs = np.nonzero(match1 * match2)[0]

        if len(idxs) < 8:
            continue

        kpts1 = pts1[match1[idxs]]
        kpts2 = pts2[match2[idxs]]

        X = LinearTriangulation(K, *extrinsic1, *extrinsic2, kpts1, kpts2)
        X = X / X[:, 3].reshape(-1, 1)

        X = NonLinearTriangulation(K, kpts1, kpts2, X, *extrinsic1[::-1], *extrinsic2[::-1])
        X = X / X[:, 3].reshape(-1, 1)

        pts3d.append(X[:, :3])
        pts2d.append(anchor_kpts[idxs])

        # print(kpts1)
        # print(kpts2)

    pts3d = np.vstack(pts3d)
    pts2d = np.vstack(pts2d)

    Rf, Tf = PnPRANSAC(K, pts2d, pts3d)

    # Rf, Tf = NonLinearPnP(K, pts2d, pts3d, R0, T0)

    print(Rf, Tf)

    print(R.from_matrix(Rf).as_rotvec(degrees=True))
    print(np.linalg.norm(Tf))



    # print(anchor_kpts)
    # print(kpts)
    # print(matches)
