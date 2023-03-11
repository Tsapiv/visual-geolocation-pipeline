import json
import os
from itertools import combinations
from pprint import pprint
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np
import open3d as o3d
from pyproj import Geod
from scipy.spatial.transform import Rotation

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


def recover_camera_position(R, T):
    C = -R.T @ T
    return C


def get_relative_C(lat1, lng1, heading1_deg, lat2, lng2, heading2_deg):
    Y1 = elevation[(lat1, lng1)]
    Y2 = elevation.get((lat2, lng2), Y1)

    azimuths_deg, _, dist = geod.inv(lng1, lat1, lng2, lat2)

    angel = np.deg2rad(90 - (azimuths_deg - heading1_deg))

    C12 = np.array([np.cos(angel) * dist, -(Y2 - Y1), np.sin(angel) * dist])
    return C12


def get_relative_E(lat1, lng1, heading1_deg, lat2, lng2, heading2_deg):
    C12 = get_relative_C(lat1, lng1, heading1_deg, lat2, lng2, heading2_deg)

    R12 = Rotation.from_rotvec(np.array([0, np.deg2rad(-(heading2_deg - heading1_deg)), 0])).as_matrix()

    T12 = -R12 @ C12

    return T12, R12


def display_pcd(pcd: Union[np.ndarray, o3d.geometry.PointCloud], cameras, reference_camera):
    if isinstance(pcd, np.ndarray):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
    for g in draw_camera_setup(cameras, reference_camera):
        vis.add_geometry(g)
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
                   distance_thr: float = 100,
                   reference_idx: int = None):
    confidences = np.asarray(confidences)
    if reference_idx is None:
        nonzero_confidence_indexes = np.nonzero(np.sum(confidences > 0.1, axis=0))[0]
        max_confidence_distribution = np.argmax(confidences, axis=0)
        most_confident_pose_idx = np.argmax(np.bincount(max_confidence_distribution[nonzero_confidence_indexes]))
    else:
        most_confident_pose_idx = reference_idx

    base_pose = camera_poses_g[most_confident_pose_idx]
    print(f'Base pose: {base_pose}')
    pts_bitmap = np.zeros(len(kpt_q), dtype=bool)

    pts3d = np.zeros((len(kpt_q), 3), dtype=np.float32)
    pts2d = np.zeros((len(kpt_q), 2), dtype=np.float32)

    pts_confidence = np.ones(len(kpt_q)) * confidence_thr

    for idx1, idx2 in combinations(range(len(matches)), 2):
        match1, confidence1, pts1 = matches[idx1], confidences[idx1], kpts_g[idx1]
        T1, R1 = get_relative_E(*base_pose, *camera_poses_g[idx1])

        match2, confidence2, pts2 = matches[idx2], confidences[idx2], kpts_g[idx2]
        T2, R2 = get_relative_E(*base_pose, *camera_poses_g[idx2])

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

    display_pcd(pts3d, camera_poses_g, base_pose)

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


def draw_camera_setup(absolute_poses, reference_pose):
    cameras = []
    print(f'Reference pose: {reference_pose}')
    for abs_pose in absolute_poses:
        Tn, Rn = get_relative_E(*reference_pose, *abs_pose)
        Tnf = np.eye(4)
        Tnf[:3, :3] = Rn
        Tnf[:3, -1] = Tn
        geometry = o3d.geometry.LineSet().create_camera_visualization(640, 640, get_K(640, 640, 90), Tnf)
        if reference_pose == abs_pose:
            geometry.paint_uniform_color((1, 0, 0))
        cameras.append(geometry)
    return cameras


PATH = 'matches6'


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
            # camera_pose_q = (49.84343056, 24.02655000, None)
            # camera_pose_q = (*camera_pose_q[:-1], None)
        except Exception as e:
            pass

        data = np.load(dir_entry.path)

        kpt_q = data['keypoints0']
        kpts_g.append(data['keypoints1'])

        confidences.append(data['match_confidence'])
        matches.append(data['matches'])
        Ks_g.append(get_K(640, 640, 90))
    pprint(camera_poses_g)

    for i in range(len(matches)):
        res = calculate_pose(matches, confidences, kpts_g, camera_poses_g, Ks_g,
                             kpt_q, K_q,
                             confidence_thr=0.2, distance_thr=200, reference_idx=i)
        if res is None:
            print('Fail')
        else:
            (T0, R0), base_camera_pose = res
            C0 = recover_camera_position(R0, T0)
            if camera_pose_q is not None:
                C1 = get_relative_C(*base_camera_pose, *camera_pose_q)
                # T1, R1 = get_relative_pose(*base_camera_pose, *camera_pose_q)
                # R0 = Rotation.from_matrix(R0).as_rotvec(degrees=True)
                # R1 = Rotation.from_matrix(R1).as_rotvec(degrees=True)

                # print(f'T0: {T0}')
                # print(f'T1: {T1}')
                # print(f'R0: {R0}')
                # print(f'R1: {R1}')
                # print(f'Err: {np.linalg.norm(T0 - T1)}m')
                # print(f'Err: {R0 - R1}deg')
                print(f'C0: {C0}')
                print(f'C1: {C1}')
                print(f'Err: {np.linalg.norm((C0 - C1)[::2])}m')
            else:
                print(f'T: {T0}')
                print(f'R: {Rotation.from_matrix(R0).as_rotvec(degrees=True)}')
                print(f'Err: {np.linalg.norm(T0)}m')


if __name__ == '__main__':
    main()
