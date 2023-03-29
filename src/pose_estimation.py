from copy import deepcopy
from itertools import combinations
from typing import List, Union

import cv2
import numpy as np
import open3d as o3d

from camera import CameraExtrinsic, Camera, CameraIntrinsic


def get_projection_matrix(intrinsic: CameraIntrinsic, extrinsic: CameraExtrinsic):
    return np.concatenate((np.dot(intrinsic.K, extrinsic.R),
                           np.dot(intrinsic.K, extrinsic.T).reshape(-1, 1)), axis=1)


def get_relative_pose(cam1: CameraExtrinsic, cam2: CameraExtrinsic) -> CameraExtrinsic:
    R12 = np.dot(cam2.R, cam1.R.T)
    T12 = -np.dot(R12, cam1.T) + cam2.T
    return CameraExtrinsic(R=R12, T=T12)


def draw_camera(camera: Camera):
    geometry = o3d.geometry.LineSet().create_camera_visualization(int(camera.metadata.w), int(camera.metadata.h), camera.intrinsic.K,
                                                                  camera.extrinsic.E)
    return geometry


def visualize_scene(pcd: Union[np.ndarray, o3d.geometry.PointCloud], cameras: List[Camera], reference_camera: Camera):
    if isinstance(pcd, np.ndarray):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    # vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
    for camera in cameras:
        camera_geometry = draw_camera(camera)
        if np.allclose(camera.extrinsic.E, reference_camera.extrinsic.E):
            camera_geometry.paint_uniform_color((1, 0, 0))
        vis.add_geometry(camera_geometry)
    ctr: o3d.visualization.ViewControl = vis.get_view_control()
    ctr.change_field_of_view(step=90)
    par: o3d.camera.PinholeCameraParameters = ctr.convert_to_pinhole_camera_parameters()
    par.extrinsic = np.eye(4)
    ctr.convert_from_pinhole_camera_parameters(par)
    vis.run()
    vis.destroy_window()


def calculate_pose(matches: List[np.ndarray],
                   confidences: List[np.ndarray],
                   kpts: List[np.ndarray],
                   cameras: List[Camera],
                   kpt: np.ndarray,
                   camera: Camera,
                   confidence_thr: float = 0.2,
                   distance_thr: float = 200,
                   reference_idx: int = None,
                   verbose: bool = False):
    confidences = np.asarray(confidences)
    if reference_idx is None:
        nonzero_confidence_indexes = np.nonzero(np.sum(confidences > 0.1, axis=0))[0]
        max_confidence_distribution = np.argmax(confidences, axis=0)
        most_confident_pose_idx = np.argmax(np.bincount(max_confidence_distribution[nonzero_confidence_indexes]))
    else:
        most_confident_pose_idx = reference_idx

    base_camera = cameras[most_confident_pose_idx]
    verbose and print(f'Base pose: {base_camera}')
    pts_bitmap = np.zeros(len(kpt), dtype=bool)

    pts3d = np.zeros((len(kpt), 3), dtype=np.float32)
    pts2d = np.zeros((len(kpt), 2), dtype=np.float32)

    pts_confidence = np.ones(len(kpt)) * confidence_thr

    for idx1, idx2 in combinations(range(len(matches)), 2):
        match1, confidence1, pts1 = matches[idx1], confidences[idx1], kpts[idx1]
        pose1 = cameras[idx1].extrinsic #get_relative_pose(base_camera.extrinsic, cameras[idx1].extrinsic)

        match2, confidence2, pts2 = matches[idx2], confidences[idx2], kpts[idx2]
        pose2 = cameras[idx2].extrinsic #get_relative_pose(base_camera.extrinsic, cameras[idx2].extrinsic)

        if np.allclose(pose1.T, pose2.T):
            verbose and print('skipping points from same pose')
            continue

        combined_confidence = confidence1 * confidence2
        confidence_mask = combined_confidence > pts_confidence
        idxs = np.nonzero(confidence_mask)[0]

        np.put(pts_confidence, idxs, combined_confidence[idxs])

        if len(idxs) < 1:
            verbose and print('skipping too few points')
            continue

        kpts1 = pts1[match1[idxs]]
        kpts2 = pts2[match2[idxs]]

        P1 = get_projection_matrix(cameras[idx1].intrinsic, pose1)
        P2 = get_projection_matrix(cameras[idx2].intrinsic, pose2)

        X = cv2.triangulatePoints(P1, P2, kpts1.T, kpts2.T).T
        X = X / X[:, 3].reshape(-1, 1)

        distance_outlier_mask1 = np.linalg.norm(X[:, :3] - cameras[idx1].extrinsic.C, axis=-1) <= distance_thr
        distance_outlier_mask2 = np.linalg.norm(X[:, :3] - cameras[idx2].extrinsic.C, axis=-1) <= distance_thr
        distance_outlier_mask = np.bitwise_or(distance_outlier_mask1, distance_outlier_mask2)

        pts_bitmap[idxs[distance_outlier_mask]] = True
        pts3d[idxs[distance_outlier_mask]] = (X[distance_outlier_mask])[:, :3]
        pts2d[idxs[distance_outlier_mask]] = kpt[idxs[distance_outlier_mask]]

    pts3d = pts3d[pts_bitmap]
    pts2d = pts2d[pts_bitmap]

    if len(pts3d) < 4:
        print('Too few 3d points')
        return

    verbose and visualize_scene(pts3d, cameras, base_camera)

    success, R_vec, t, inliers = cv2.solvePnPRansac(objectPoints=pts3d, imagePoints=pts2d,
                                                    cameraMatrix=camera.intrinsic.K,
                                                    distCoeffs=camera.intrinsic.distortion_coefficients,
                                                    flags=cv2.SOLVEPNP_EPNP, confidence=0.999999,
                                                    reprojectionError=5, iterationsCount=10000)
    if not success:
        print('Ransac failed')
        return

    R, _ = cv2.Rodrigues(R_vec)
    T = t[:, 0]

    estimated_camera = deepcopy(camera)

    estimated_camera.extrinsic = CameraExtrinsic(R=R, T=T)

    return estimated_camera, base_camera
