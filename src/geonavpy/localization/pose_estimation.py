from copy import deepcopy
from enum import Enum, auto
from itertools import combinations
from typing import List

import cv2
import numpy as np

from .visualization import visualize_reconstruction
from ..common.camera import CameraExtrinsic, Camera, CameraIntrinsic


class ReconstructionPolicy(Enum):
    EXPANSION = auto()
    EXPANSION2 = auto()
    DISPLACEMENT = auto()


RP = ReconstructionPolicy


def _get_projection_matrix(intrinsic: CameraIntrinsic, extrinsic: CameraExtrinsic):
    return np.concatenate((np.dot(intrinsic.K, extrinsic.R),
                           np.dot(intrinsic.K, extrinsic.T).reshape(-1, 1)), axis=1)


def _get_relative_pose(cam1: CameraExtrinsic, cam2: CameraExtrinsic) -> CameraExtrinsic:
    R12 = np.dot(cam2.R, cam1.R.T)
    T12 = -np.dot(R12, cam1.T) + cam2.T
    return CameraExtrinsic(R=R12, T=T12)


def _compute_2_view_pointcloud(kpts1: np.ndarray,
                               kpts2: np.ndarray,
                               cam1: Camera,
                               cam2: Camera,
                               matches: np.ndarray,
                               match_confidence: np.ndarray,
                               match_confidence_thr: float,
                               distance_thr: float,
                               verbose: bool = False):
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
    verbose and print(
        f'err: {np.linalg.norm(cam1.extrinsic.C - cam1_extrinsic.C)} C: {cam1_extrinsic.C} GT: {cam1.extrinsic.C} diff: {cam1_extrinsic.C - cam1.extrinsic.C}')

    P1 = _get_projection_matrix(cam1.intrinsic, cam1_extrinsic)
    P2 = _get_projection_matrix(cam2.intrinsic, cam2.extrinsic)

    X = cv2.triangulatePoints(P1, P2, mkpts1.T, mkpts2.T).T
    X = X / X[:, 3].reshape(-1, 1)
    distance_outlier_mask = np.linalg.norm(X[:, :3] - cam2.extrinsic.C, axis=-1) <= distance_thr

    return X[distance_outlier_mask][:, :3], mkpts1[distance_outlier_mask]


def estimate_pose(matches: List[np.ndarray],
                  confidences: List[np.ndarray],
                  kpts: List[np.ndarray],
                  cameras: List[Camera],
                  kpt: np.ndarray,
                  camera: Camera,
                  policy: RP = RP.DISPLACEMENT,
                  confidence_thr: float = 0.2,
                  distance_thr: float = 200,
                  verbose: bool = False):
    confidences = np.asarray(confidences)
    pts_bitmap = np.zeros(len(kpt), dtype=bool)

    if policy == RP.DISPLACEMENT:
        pts3d = np.zeros((len(kpt), 3), dtype=np.float32)
        pts2d = np.zeros((len(kpt), 2), dtype=np.float32)
    else:
        pts3d = []
        pts2d = []

    pts_confidence = np.ones(len(kpt)) * confidence_thr

    if policy == RP.EXPANSION2:
        for idx in range(len(matches)):
            ret = _compute_2_view_pointcloud(kpt, kpts[idx],
                                             camera,
                                             cameras[idx],
                                             matches[idx],
                                             confidences[idx],
                                             confidence_thr,
                                             distance_thr,
                                             verbose)
            if ret is None:
                continue
            pts3d.extend(ret[0])
            pts2d.extend(ret[1])

    for idx1, idx2 in combinations(range(len(matches)), 2):
        match1, confidence1, pts1 = matches[idx1], confidences[idx1], kpts[idx1]
        pose1 = cameras[idx1].extrinsic

        match2, confidence2, pts2 = matches[idx2], confidences[idx2], kpts[idx2]
        pose2 = cameras[idx2].extrinsic

        if np.allclose(pose1.C, pose2.C):
            verbose and print('Skipping points from same pose')
            continue

        combined_confidence = confidence1 * confidence2
        confidence_mask = combined_confidence > pts_confidence
        idxs = np.nonzero(confidence_mask)[0]

        if policy == RP.DISPLACEMENT:
            np.put(pts_confidence, idxs, combined_confidence[idxs])

        if len(idxs) < 1:
            verbose and print('Skipping too few points')
            continue

        kpts1 = pts1[match1[idxs]]
        kpts2 = pts2[match2[idxs]]

        P1 = _get_projection_matrix(cameras[idx1].intrinsic, pose1)
        P2 = _get_projection_matrix(cameras[idx2].intrinsic, pose2)

        kpts1 = np.squeeze(cv2.undistortImagePoints(kpts1.T, cameraMatrix=cameras[idx1].intrinsic.K,
                                                    distCoeffs=cameras[idx1].intrinsic.distortion_coefficients))
        kpts2 = np.squeeze(cv2.undistortImagePoints(kpts2.T, cameraMatrix=cameras[idx2].intrinsic.K,
                                                    distCoeffs=cameras[idx2].intrinsic.distortion_coefficients))

        X = cv2.triangulatePoints(P1, P2, kpts1.T, kpts2.T).T
        X = X / X[:, 3].reshape(-1, 1)

        distance_outlier_mask1 = np.linalg.norm(X[:, :3] - cameras[idx1].extrinsic.C, axis=-1) <= distance_thr
        distance_outlier_mask2 = np.linalg.norm(X[:, :3] - cameras[idx2].extrinsic.C, axis=-1) <= distance_thr
        distance_outlier_mask = np.bitwise_or(distance_outlier_mask1, distance_outlier_mask2)

        if policy == RP.DISPLACEMENT:
            pts_bitmap[idxs[distance_outlier_mask]] = True
            pts3d[idxs[distance_outlier_mask]] = (X[distance_outlier_mask])[:, :3]
            pts2d[idxs[distance_outlier_mask]] = kpt[idxs[distance_outlier_mask]]
        else:
            pts3d.extend((X[distance_outlier_mask])[:, :3])
            pts2d.extend(kpt[idxs[distance_outlier_mask]])

    if policy == RP.DISPLACEMENT:
        pts3d = pts3d[pts_bitmap]
        pts2d = pts2d[pts_bitmap]
    else:
        pts3d = np.asarray(pts3d)
        pts2d = np.asarray(pts2d)

    if len(pts3d) < 4:
        verbose and print('Too few 3d points')
        return

    success, R_vec, t, inliers = cv2.solvePnPRansac(objectPoints=pts3d, imagePoints=pts2d,
                                                    cameraMatrix=camera.intrinsic.K,
                                                    distCoeffs=camera.intrinsic.distortion_coefficients,
                                                    flags=cv2.SOLVEPNP_EPNP, confidence=0.999999,
                                                    reprojectionError=1, iterationsCount=10000)
    if not success:
        verbose and visualize_reconstruction(pts3d, cameras, camera, None)
        verbose and print('Ransac failed')
        return

    R, _ = cv2.Rodrigues(R_vec)
    T = t[:, 0]

    estimated_camera = deepcopy(camera)

    estimated_camera.extrinsic = CameraExtrinsic(R=R, T=T)

    verbose and visualize_reconstruction(pts3d, cameras, camera, estimated_camera)

    return estimated_camera
