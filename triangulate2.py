import json
import os
from itertools import combinations
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np
import open3d as o3d
from pyproj import Geod, Proj
from scipy.spatial.transform import Rotation as R

WGS84_a = 6378137.0
WGS84_b = 6356752.314245


def ecef_from_lla(lat, lon, alt: float) -> Tuple[float, ...]:
    """
    Compute ECEF XYZ from latitude, longitude and altitude.

    All using the WGS84 model.
    Altitude is the distance to the WGS84 ellipsoid.
    Check results here http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
    """
    a2 = WGS84_a ** 2
    b2 = WGS84_b ** 2
    lat = np.radians(lat)
    lon = np.radians(lon)
    L = 1.0 / np.sqrt(a2 * np.cos(lat) ** 2 + b2 * np.sin(lat) ** 2)
    x = (a2 * L + alt) * np.cos(lat) * np.cos(lon)
    y = (a2 * L + alt) * np.cos(lat) * np.sin(lon)
    z = (b2 * L + alt) * np.sin(lat)
    return x, y, z

def ecef_from_topocentric_transform(lat, lon, alt: float) -> np.ndarray:
    """
    Transformation from a topocentric frame at reference position to ECEF.

    The topocentric reference frame is a metric one with the origin
    at the given (lat, lon, alt) position, with the X axis heading east,
    the Y axis heading north and the Z axis vertical to the ellipsoid.
    """
    x, y, z = ecef_from_lla(lat, lon, alt)
    sa = np.sin(np.radians(lat))
    ca = np.cos(np.radians(lat))
    so = np.sin(np.radians(lon))
    co = np.cos(np.radians(lon))
    return np.array(
        [
            [-so, -sa * co, ca * co, x],
            [co, -sa * so, ca * so, y],
            [0, ca, sa, z],
            [0, 0, 0, 1],
        ]
    )

def topocentric_from_lla(lat, lon, alt: float, reflat, reflon, refalt: float):
    """
    Transform from lat, lon, alt to topocentric XYZ.
    """
    T = np.linalg.inv(ecef_from_topocentric_transform(reflat, reflon, refalt))
    x, y, z = ecef_from_lla(lat, lon, alt)
    tx = T[0, 0] * x + T[0, 1] * y + T[0, 2] * z + T[0, 3]
    ty = T[1, 0] * x + T[1, 1] * y + T[1, 2] * z + T[1, 3]
    tz = T[2, 0] * x + T[2, 1] * y + T[2, 2] * z + T[2, 3]
    return tx, ty, tz


proj = Proj(proj='utm', zone=35, ellps='WGS84')

geod = Geod(ellps='WGS84')


def convert_elevation(inp):
    output = {}
    for entry in inp:
        output[(entry['location']['lat'], entry['location']['lng'])] = entry['elevation']
    return output


elevation = convert_elevation(json.load(open('data/49.8443931@24.0254815/elevation.json')))


def get_K(w, h, fov):
    K = np.array([[w / np.tan(np.deg2rad(fov / 2)) / 2, 0, w / 2],
                  [0, h / np.tan(np.deg2rad(fov / 2)) / 2, h / 2],
                  [0, 0, 1]])
    return K


def get_projection_matrix(K, R, T):
    return np.concatenate((np.dot(K, R), np.dot(K, T).reshape(-1, 1)), axis=1)


def parse_camera_pose(name: str):
    coords, heading = name.split('_')
    heading = float(heading)
    lat, lng = list(map(float, coords.split('@')))
    return lat, lng, heading


def get_relative_pose(lat1, lng1, heading1_deg, lat2, lng2, heading2_deg):
    Y1 = elevation[(lat1, lng1)]
    Y2 = elevation[(lat2, lng2)]

    X1, Z1 = proj(lng1, lat1)
    X2, Z2 = proj(lng2, lat2)

    # X1, Z1 = proj(lat1, lng1)
    # X2, Z2 = proj(lat2, lng2)
    # change_sign = -1 if heading1_deg > 180 else 1  # for some reason if base angle 270 there is no need to negate but if base angle is 90 then sign needs to be changed
    T12 = np.array([X2 - X1, (-Y2 + Y1), Z2 - Z1])[::-1]

    # azimuths_deg, _, dist = geod.inv(lng1, lat1, lng2, lat2)
    #
    # T12 = np.array([np.sin(np.deg2rad(azimuths_deg)) * dist, 0, np.cos(np.deg2rad(azimuths_deg)) * dist])[::1]
    # T12[1] = (Y2 - Y1)

    R12 = R.from_rotvec(np.array([0, -np.deg2rad((heading2_deg - heading1_deg) % 360), 0])).as_matrix()

    return T12, R12

# def get_relative_pose(lat1, lng1, heading1_deg, lat2, lng2, heading2_deg):
#     alt1 = elevation[(lat1, lng1)]
#     alt2 = elevation.get((lat2, lng2), alt1)
#
#     X, Y, Z = topocentric_from_lla(lat2, lng2, alt2, lat1, lng1, alt1)
#
#     T12 = np.asarray([X, alt1 - alt2, Y])
#
#
#     R12 = R.from_rotvec(np.array([0, -np.deg2rad(heading2_deg), 0])).as_matrix()
#
#     return T12, R12


def get_transformation(R, T):
    tform = np.eye(4)
    tform[:3, :3] = R
    tform[:3, -1] = T
    return tform


def display_pcd(pcd: Union[np.ndarray, o3d.geometry.PointCloud]):
    if isinstance(pcd, np.ndarray):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
    ctr: o3d.visualization.ViewControl = vis.get_view_control()
    ctr.change_field_of_view(step=90)
    par: o3d.camera.PinholeCameraParameters = ctr.convert_to_pinhole_camera_parameters()
    par.extrinsic = np.eye(4)
    ctr.convert_from_pinhole_camera_parameters(par)
    vis.run()
    vis.destroy_window()



def calculate_pose(matches: List[np.ndarray],
                   confidences: List[np.ndarray],
                   kpts_g: List[np.ndarray],
                   camera_poses_g: List[Tuple],
                   Ks_g: List[np.ndarray],
                   kpt_q: np.ndarray,
                   K_q: Optional[np.ndarray] = None,
                   confidence_thr: float = 0.89,
                   distance_thr: float = 100):
    confidences = np.asarray(confidences)

    nonzero_confidence_indexes = np.nonzero(np.sum(confidences > 0.15, axis=0))[0]
    max_confidence_distribution = np.argmax(confidences, axis=0)
    most_confident_pose_idx = np.argmax(np.bincount(max_confidence_distribution[nonzero_confidence_indexes]))

    base_pose = camera_poses_g[most_confident_pose_idx]
    print(f'Base pose: {base_pose}')
    pts_bitmap = np.zeros(len(kpt_q), dtype=bool)

    pts3d = np.zeros((len(kpt_q), 3), dtype=np.float32)
    pts2d = np.zeros((len(kpt_q), 2), dtype=np.float32)

    pts_confidence = np.ones(len(kpt_q)) * confidence_thr

    for idx1, idx2 in combinations(range(len(matches)), 2):
        match1, confidence1, pts1 = matches[idx1], confidences[idx1], kpts_g[idx1]
        T1, R1 = get_relative_pose(*base_pose, *camera_poses_g[idx1])

        match2, confidence2, pts2 = matches[idx2], confidences[idx2], kpts_g[idx2]
        T2, R2 = get_relative_pose(*base_pose, *camera_poses_g[idx2])

        if np.allclose(T1, T2):
            print('skipping points from same pose')
            continue

        combined_confidence = confidence1 * confidence2
        confidence_mask = combined_confidence > pts_confidence
        idxs = np.nonzero(confidence_mask)[0]

        np.put(pts_confidence, idxs, combined_confidence[idxs])

        if len(idxs) < 1:
            print('skipping too few points')
            continue

        kpts1 = pts1[match1[idxs]]
        kpts2 = pts2[match2[idxs]]

        P1 = get_projection_matrix(Ks_g[idx1], R1, T1)
        P2 = get_projection_matrix(Ks_g[idx2], R2, T2)

        X = cv2.triangulatePoints(P1, P2, kpts1.T, kpts2.T).T
        X = X / X[:, 3].reshape(-1, 1)

        distance_outlier_mask = np.linalg.norm(X[:, :3], axis=-1) <= distance_thr

        pts_bitmap[idxs[distance_outlier_mask]] = True
        pts3d[idxs[distance_outlier_mask]] = (X[distance_outlier_mask])[:, :3]
        pts2d[idxs[distance_outlier_mask]] = kpt_q[idxs[distance_outlier_mask]]

    pts3d = pts3d[pts_bitmap]
    pts2d = pts2d[pts_bitmap]

    if len(pts3d) < 4:
        print('Too few 3d points')
        return

    display_pcd(pts3d)

    success, R_vec, t, inliers = cv2.solvePnPRansac(objectPoints=pts3d, imagePoints=pts2d, cameraMatrix=K_q,
                                                    distCoeffs=None,
                                                    flags=cv2.SOLVEPNP_EPNP, confidence=0.999999,
                                                    reprojectionError=5, iterationsCount=10000)
    if not success:
        print('Ransac failed')
        return

    # R_vec, t = cv2.solvePnPRefineLM(objectPoints=pts3d, imagePoints=pts2d, cameraMatrix=K_q,
    #                                                 distCoeffs=None, rvec=R_vec, tvec=t)

    R0, _ = cv2.Rodrigues(R_vec)
    T0 = t[:, 0]

    return (T0, R0), base_pose

PATH = 'matches5'

def main():
    matches = []
    confidences = []
    kpts_g = []
    camera_poses_g = []
    Ks_g = []
    kpt_q = None
    camera_pose_q = None
    K_q = get_K(640, 640, 90)

    for dir_entry in os.scandir(PATH):
        if not dir_entry.name.endswith('npz'):
            continue
        match_name = dir_entry.name.strip('_matches.npz')
        match_name1, match_name2 = '_'.join(match_name.split('_')[:-2]), '_'.join(match_name.split('_')[-2:])

        camera_poses_g.append(parse_camera_pose(match_name2))

        try:
            camera_pose_q = parse_camera_pose(match_name1)
        except Exception as e:
            pass

        data = np.load(dir_entry.path)

        kpt_q = data['keypoints0']
        kpts_g.append(data['keypoints1'])

        confidences.append(data['match_confidence'])
        matches.append(data['matches'])
        Ks_g.append(get_K(640, 640, 90))

    res = calculate_pose(matches, confidences, kpts_g, camera_poses_g, Ks_g,
                         kpt_q, K_q,
                         confidence_thr=0.5, distance_thr=150)
    if res is None:
        print('Fail')
    else:
        (T0, R0), base_camera_pose = res
        if camera_pose_q is not None:
            T1, R1 = get_relative_pose(*base_camera_pose, *camera_pose_q)
            R0 = R.from_matrix(R0).as_rotvec(degrees=True)
            R1 = R.from_matrix(R1).as_rotvec(degrees=True)

            print(f'T0: {T0}')
            print(f'T1: {T1}')
            print(f'R0: {R0}')
            print(f'R1: {R1}')
            print(f'Err: {np.linalg.norm(T0 - T1)}m')
            print(f'Err: {R0 - R1}deg')
        else:
            print(f'T: {T0}')
            print(f'R: {R.from_matrix(R0).as_rotvec(degrees=True)}')
            print(f'Err: {np.linalg.norm(T0)}m')


if __name__ == '__main__':
    main()
