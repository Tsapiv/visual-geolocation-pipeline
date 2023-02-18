import json
import os
from itertools import combinations

import cv2
import open3d as o3d
import numpy as np
from pyproj import Geod, Proj
from scipy.spatial.transform import Rotation as R

from position_refinement.LinearTriangulation import LinearTriangulation
from position_refinement.NonLinearTriangulation import NonLinearTriangulation

PATH = 'matches2'

proj = Proj(proj='utm', zone=35, ellps='WGS84')

geod = Geod(ellps='WGS84')



def convert_elevation(inp):
    output = {}
    for entry in inp:
        output[(entry['location']['lat'], entry['location']['lng'])] = entry['elevation']
    return output

elevation = convert_elevation(json.load(open('data/49.8443931@24.0254815/elevation.json')))

w, h = 640, 640
fov = 90

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
    change_sign = -1 if heading1_deg > 180 else 1 # for some reason if base angle 270 there is no need to negate but if base angle is 90 then sign needs to be changed
    T12 = np.array([X2 - X1, Y2 - Y1, Z2 - Z1])[::-1] * change_sign

    R12 = R.from_rotvec(np.array([0, -np.deg2rad(heading2_deg - heading1_deg), 0])).as_matrix()

    return T12, R12


def get_relative_pose(R1, T1, R2, T2):
    R12 = np.dot(R2, R1.T)
    T12 = -np.dot(R12, T1) + T2
    return R12, T12

def get_transformation(R, T):
    tform = np.eye(4)
    tform[:3, :3] = R
    tform[:3, -1] = T
    return tform

def display_pcd(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
    ctr: o3d.visualization.ViewControl = vis.get_view_control()
    print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
    ctr.change_field_of_view(step=90)
    ctr.set_lookat(np.asarray([1, 0, 0]))
    par: o3d.camera.PinholeCameraParameters = ctr.convert_to_pinhole_camera_parameters()
    tmp = np.copy(par.extrinsic)
    tmp[:3, -1] = 0
    par.extrinsic = np.eye(4)
    ctr.convert_from_pinhole_camera_parameters(par)
    # ctr.set_lookat(np.asarray([1, 0, 0]))
    par: o3d.camera.PinholeCameraParameters = ctr.convert_to_pinhole_camera_parameters()

    print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':

    archives = []
    extrinsics = []
    for dir_entry in os.scandir(PATH):
        if not dir_entry.name.endswith('npz'):
            continue
        archives.append(np.load(dir_entry.path))
        extrinsics.append(parse_relative_motion(dir_entry.name))
        print(extrinsics[-1][0])

    anchor_kpts = archives[0]['keypoints0']

    kpts = []
    matches = []
    confidences = []
    common_matches = np.ones_like(archives[0]['matches'])
    for arch in archives:
        kpts.append(arch['keypoints1'])
        confidences.append(arch['match_confidence'])
        matches.append(arch['matches'])

    pts3d = np.ones((len(anchor_kpts), 3)) * -999
    pts2d = np.ones((len(anchor_kpts), 2)) * -999

    pts_confidence = np.ones(len(anchor_kpts)) * 0.89

    for idx1, idx2 in combinations(range(len(matches)), 2):
        match1, confidence1, pts1, (T1, R1) = matches[idx1], confidences[idx1], kpts[idx1], extrinsics[idx1]
        match2, confidence2, pts2, (T2, R2) = matches[idx2], confidences[idx2], kpts[idx2], extrinsics[idx2]
        # match3, pts3, (T3, R3) = matches[idx3], kpts[idx3], extrinsics[idx3]
        # match4 = matches[idx4]
        # match5 = matches[idx5]

        combined_confidence = confidence1 * confidence2
        confidence_mask = combined_confidence > pts_confidence
        idxs = np.nonzero(confidence_mask)[0]

        np.put(pts_confidence, idxs, combined_confidence[idxs])


        if len(idxs) < 1:
            print('skipping too few points')
            continue
        print(len(idxs))

        if np.allclose(T1, T2):
            print('skipping points from same pose')
            continue

        kpts1 = pts1[match1[idxs]]
        kpts2 = pts2[match2[idxs]]

        # R12, T12 = get_relative_pose(R1, T1, R2, T2)

        P1 = get_projection_matrix(K, R1, T1)
        P2 = get_projection_matrix(K, R2, T2)

        X = cv2.triangulatePoints(P1, P2, kpts1.T, kpts2.T).T
        X = X / X[:, 3].reshape(-1, 1)

        # tform = np.linalg.inv(get_transformation(R1, T1))
        #
        # X = (tform @ X.T).T

        distance_outlier_mask = np.linalg.norm(X[:, :3], axis=-1) <= 100

        # pts3d.append((X[distance_outlier_mask])[:, :3])
        # pts2d.append((anchor_kpts[idxs])[distance_outlier_mask])

        pts3d[idxs[distance_outlier_mask]] = (X[distance_outlier_mask])[:, :3]
        pts2d[idxs[distance_outlier_mask]] = anchor_kpts[idxs[distance_outlier_mask]]

        # display_pcd(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts3d[-1])))

        # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts3d[-1]))
        # o3d.visualization.draw_geometries([pcd])
        #
        # if len(pts3d[-1]) < 4:
        #     continue
        #
        # print('='*20)
        # print('match', idx1, idx2)
        # success, R_vec, t, inliers = cv2.solvePnPRansac(objectPoints=pts3d[-1], imagePoints=pts2d[-1], cameraMatrix=K,
        #                                                 distCoeffs=None,
        #                                                 flags=cv2.SOLVEPNP_EPNP, confidence=0.999999,
        #                                                 reprojectionError=1, iterationsCount=20000)
        #
        # R0, _ = cv2.Rodrigues(R_vec)
        # T0 = t[:, 0]
        #
        # print(success)
        # print(R0)
        # print(T0)
        # print(R.from_matrix(R0).as_rotvec(degrees=True))
        # print(np.linalg.norm(T0))

        # print(kpts1)
        # print(kpts2)



    # pts3d = np.vstack(pts3d)
    # pts2d = np.vstack(pts2d)

    unset_mask = np.all(pts3d != -999, axis=-1)

    pts3d = pts3d[unset_mask]
    pts2d = pts2d[unset_mask]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts3d))

    display_pcd(pcd)
    #
    # o3d.visualization.draw_geometries([pcd])

    success, R_vec, t, inliers = cv2.solvePnPRansac(objectPoints=pts3d, imagePoints=pts2d, cameraMatrix=K,
                                                    distCoeffs=None,
                                                    flags=cv2.SOLVEPNP_EPNP, confidence=0.999999,
                                                    reprojectionError=0.5, iterationsCount=10000)

    # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts3d[np.squeeze(inliers)]))
    #
    # o3d.io.write_point_cloud('pcd2.ply', pcd)

    # R_vec, t = cv2.solvePnPRefineLM(objectPoints=pts3d, imagePoints=pts2d, cameraMatrix=K,
    #                                                 distCoeffs=None, rvec=R_vec, tvec=t)

    R0, _ = cv2.Rodrigues(R_vec)
    T0 = t[:, 0]

    print(len(inliers))
    print(success)
    print(R0)
    print(T0)

    # print(R0, T0)
    print(R.from_matrix(R0).as_rotvec(degrees=True))
    print(np.linalg.norm(T0))

    # print(anchor_kpts)
    # print(kpts)
    # print(matches)
