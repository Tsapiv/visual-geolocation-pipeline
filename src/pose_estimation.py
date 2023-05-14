from copy import deepcopy
from enum import Enum, auto
from itertools import combinations
from typing import List, Union

import cv2
import numpy as np
import open3d as o3d

from camera import CameraExtrinsic, Camera, CameraIntrinsic


class ReconstructionPolicy(Enum):
    Expansion = auto()
    Displacement = auto()


def get_projection_matrix(intrinsic: CameraIntrinsic, extrinsic: CameraExtrinsic):
    return np.concatenate((np.dot(intrinsic.K, extrinsic.R),
                           np.dot(intrinsic.K, extrinsic.T).reshape(-1, 1)), axis=1)


def get_relative_pose(cam1: CameraExtrinsic, cam2: CameraExtrinsic) -> CameraExtrinsic:
    R12 = np.dot(cam2.R, cam1.R.T)
    T12 = -np.dot(R12, cam1.T) + cam2.T
    return CameraExtrinsic(R=R12, T=T12)


def draw_camera(camera: Camera):
    geometry = o3d.geometry.LineSet().create_camera_visualization(int(camera.metadata.w), int(camera.metadata.h),
                                                                  camera.intrinsic.K,
                                                                  camera.extrinsic.E)
    return geometry


def visualize_scene(pcd: Union[np.ndarray, o3d.geometry.PointCloud], cameras: List[Camera], reference_camera: Camera):
    if isinstance(pcd, np.ndarray):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    for camera in cameras:
        camera_geometry = draw_camera(camera)
        if np.allclose(camera.extrinsic.E, reference_camera.extrinsic.E):
            camera_geometry.paint_uniform_color((1, 0, 0))
        vis.add_geometry(camera_geometry)
    ctr: o3d.visualization.ViewControl = vis.get_view_control()
    ctr.change_field_of_view(step=90)
    par: o3d.camera.PinholeCameraParameters = ctr.convert_to_pinhole_camera_parameters()
    par.extrinsic = reference_camera.extrinsic.E
    ctr.convert_from_pinhole_camera_parameters(par)
    vis.run()
    vis.destroy_window()


def visualize_reconstruction(pcd: Union[np.ndarray, o3d.geometry.PointCloud], cameras: List[Camera], gt_camera: Camera,
                             estimated_camera: Camera):
    if isinstance(pcd, np.ndarray):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    for camera in cameras:
        if np.allclose(camera.extrinsic.C, gt_camera.extrinsic.C):
            continue
        vis.add_geometry(draw_camera(camera))

    if gt_camera.extrinsic is not None:
        gt_camera_geom = draw_camera(gt_camera)
        gt_camera_geom.paint_uniform_color((0, 1, 0))
        vis.add_geometry(gt_camera_geom)

    estimated_camera_geom = draw_camera(estimated_camera)
    estimated_camera_geom.paint_uniform_color((1, 0, 0))
    vis.add_geometry(estimated_camera_geom)

    ctr: o3d.visualization.ViewControl = vis.get_view_control()
    ctr.change_field_of_view(step=90)
    par: o3d.camera.PinholeCameraParameters = ctr.convert_to_pinhole_camera_parameters()
    par.extrinsic = estimated_camera.extrinsic.E
    ctr.convert_from_pinhole_camera_parameters(par)
    vis.run()
    vis.destroy_window()


def triangulate_with_estimated_pose(kpts1: np.ndarray,
                                    kpts2: np.ndarray,
                                    cam1: Camera,
                                    cam2: Camera,
                                    matches: np.ndarray,
                                    match_confidence: np.ndarray,
                                    match_confidence_thr: float,
                                    distance_thr: float):
    mask = match_confidence > np.sqrt(match_confidence_thr)
    matches = matches[mask]

    mkpts1 = kpts1[mask]
    mkpts2 = kpts2[matches]
    try:
        mkpts1_norm = np.ascontiguousarray(
            cv2.undistortPoints(np.expand_dims(mkpts1, axis=1), cameraMatrix=cam1.intrinsic.K,
                                distCoeffs=cam1.intrinsic.distortion_coefficients))
        mkpts2_norm = np.ascontiguousarray(
            cv2.undistortPoints(np.expand_dims(mkpts2, axis=1), cameraMatrix=cam2.intrinsic.K,
                                distCoeffs=cam2.intrinsic.distortion_coefficients))

        E, mask = cv2.findEssentialMat(mkpts1_norm, mkpts2_norm, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999,
                                       threshold=1.0)
        points, R_12, T_12, mask = cv2.recoverPose(E, mkpts1_norm, mkpts2_norm)
    except Exception:
        return
    T_12 = T_12[:, 0]

    R_1 = (cam2.extrinsic.R.T @ R_12).T
    T_1 = -R_12.T @ (T_12 - cam2.extrinsic.T)

    cam1_extrinsic = CameraExtrinsic(R_1, T_1)
    print(
        f'err: {np.linalg.norm(cam1.extrinsic.C - cam1_extrinsic.C)} C: {cam1_extrinsic.C} GT: {cam1.extrinsic.C} diff: {cam1_extrinsic.C - cam1.extrinsic.C}')

    P1 = get_projection_matrix(cam1.intrinsic, cam1_extrinsic)
    P2 = get_projection_matrix(cam2.intrinsic, cam2.extrinsic)

    X = cv2.triangulatePoints(P1, P2, mkpts1.T, mkpts2.T).T
    X = X / X[:, 3].reshape(-1, 1)
    distance_outlier_mask = np.linalg.norm(X[:, :3] - cam2.extrinsic.C, axis=-1) <= distance_thr

    return X[distance_outlier_mask][:, :3], mkpts1[distance_outlier_mask]


def calculate_pose(matches: List[np.ndarray],
                   confidences: List[np.ndarray],
                   kpts: List[np.ndarray],
                   cameras: List[Camera],
                   kpt: np.ndarray,
                   camera: Camera,
                   policy: ReconstructionPolicy = ReconstructionPolicy.Displacement,
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

    if policy == ReconstructionPolicy.Displacement:
        pts3d = np.zeros((len(kpt), 3), dtype=np.float32)
        pts2d = np.zeros((len(kpt), 2), dtype=np.float32)
    else:
        pts3d = []
        pts2d = []

    pts_confidence = np.ones(len(kpt)) * confidence_thr

    # for idx in range(len(matches)):
    #     ret = triangulate_with_estimated_pose(kpt, kpts[idx], camera, cameras[idx], matches[idx], confidences[idx], confidence_thr, distance_thr)
    #     if ret is None:
    #         continue
    #     pts3d.extend(ret[0])
    #     pts2d.extend(ret[1])

    for idx1, idx2 in combinations(range(len(matches)), 2):
        match1, confidence1, pts1 = matches[idx1], confidences[idx1], kpts[idx1]
        pose1 = cameras[idx1].extrinsic  # get_relative_pose(base_camera.extrinsic, cameras[idx1].extrinsic)

        match2, confidence2, pts2 = matches[idx2], confidences[idx2], kpts[idx2]
        pose2 = cameras[idx2].extrinsic  # get_relative_pose(base_camera.extrinsic, cameras[idx2].extrinsic)

        if np.allclose(pose1.C, pose2.C):
            verbose and print('skipping points from same pose')
            continue

        combined_confidence = confidence1 * confidence2
        confidence_mask = combined_confidence > pts_confidence
        idxs = np.nonzero(confidence_mask)[0]

        if policy == ReconstructionPolicy.Displacement:
            np.put(pts_confidence, idxs, combined_confidence[idxs])

        if len(idxs) < 1:
            verbose and print('skipping too few points')
            continue

        kpts1 = pts1[match1[idxs]]
        kpts2 = pts2[match2[idxs]]

        P1 = get_projection_matrix(cameras[idx1].intrinsic, pose1)
        P2 = get_projection_matrix(cameras[idx2].intrinsic, pose2)

        kpts1 = np.squeeze(cv2.undistortImagePoints(kpts1.T, cameraMatrix=cameras[idx1].intrinsic.K,
                                                    distCoeffs=cameras[idx1].intrinsic.distortion_coefficients))
        kpts2 = np.squeeze(cv2.undistortImagePoints(kpts2.T, cameraMatrix=cameras[idx2].intrinsic.K,
                                                    distCoeffs=cameras[idx2].intrinsic.distortion_coefficients))

        X = cv2.triangulatePoints(P1, P2, kpts1.T, kpts2.T).T
        X = X / X[:, 3].reshape(-1, 1)

        distance_outlier_mask1 = np.linalg.norm(X[:, :3] - cameras[idx1].extrinsic.C, axis=-1) <= distance_thr
        distance_outlier_mask2 = np.linalg.norm(X[:, :3] - cameras[idx2].extrinsic.C, axis=-1) <= distance_thr
        distance_outlier_mask = np.bitwise_or(distance_outlier_mask1, distance_outlier_mask2)

        if policy == ReconstructionPolicy.Displacement:
            pts_bitmap[idxs[distance_outlier_mask]] = True
            pts3d[idxs[distance_outlier_mask]] = (X[distance_outlier_mask])[:, :3]
            pts2d[idxs[distance_outlier_mask]] = kpt[idxs[distance_outlier_mask]]
        else:
            pts3d.extend((X[distance_outlier_mask])[:, :3])
            pts2d.extend(kpt[idxs[distance_outlier_mask]])

    if policy == ReconstructionPolicy.Displacement:
        pts3d = pts3d[pts_bitmap]
        pts2d = pts2d[pts_bitmap]
    else:
        pts3d = np.asarray(pts3d)
        pts2d = np.asarray(pts2d)

    if len(pts3d) < 4:
        print('Too few 3d points')
        return

    verbose and visualize_scene(pts3d, cameras, base_camera)

    success, R_vec, t, inliers = cv2.solvePnPRansac(objectPoints=pts3d, imagePoints=pts2d,
                                                    cameraMatrix=camera.intrinsic.K,
                                                    distCoeffs=camera.intrinsic.distortion_coefficients,
                                                    flags=cv2.SOLVEPNP_EPNP, confidence=0.999999,
                                                    reprojectionError=1, iterationsCount=10000)
    if not success:
        print('Ransac failed')
        return

    R, _ = cv2.Rodrigues(R_vec)
    T = t[:, 0]

    estimated_camera = deepcopy(camera)

    estimated_camera.extrinsic = CameraExtrinsic(R=R, T=T)

    verbose and visualize_reconstruction(pts3d, cameras, camera, estimated_camera)

    return estimated_camera, base_camera
