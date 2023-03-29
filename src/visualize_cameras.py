import random
from copy import deepcopy
from typing import Dict

import cv2
import open3d as o3d
import numpy as np
from tqdm import tqdm

from camera import Camera, CameraExtrinsic, CameraIntrinsic, CameraMetadata
from dataset import Dataset
from pose_estimation import calculate_pose, get_relative_pose, draw_camera
from ranking.index import Index
from matching.superpoint import SuperPoint
from matching.superglue import SuperGlue

from keypoints_generation import generate_keypoints, tensor2numpy
from keypoints_matching import match_keypoints
from geoutils import relative_extrinsic_from_metadata, intrinsics_from_metadata, extrinsic_from_metadata, \
    extrinsic_from_metadata_and_switch_axis


# intrinsics = CameraIntrinsic(K=np.asarray([[744.375, 0, 426], [0, 744.375, 240], [0, 0, 1]]))
# intrinsics = CameraIntrinsic(K=np.asarray([[744.375, 0, 0], [0, 744.375, 0], [0, 0, 1]]))


if __name__ == '__main__':

    # trainset = Dataset(root='datasets/KingsCollegeTrain', descriptor_type='radenovic_gldv1')
    testset = Dataset(root='datasets/aachen_v1_train', descriptor_type='radenovic_gldv1')

    cameras = []
    ref = None

    for entry in tqdm(testset.entries):
        # query_image = testset.image(entry)
        query_metadata = CameraMetadata(**testset.metadata(entry))

        query_camera = Camera(intrinsics_from_metadata(query_metadata), extrinsic_from_metadata(query_metadata), metadata=query_metadata)

        # if ref is None:
        #     ref = deepcopy(query_camera)
        #
        #
        #
        # query_camera.extrinsic = get_relative_pose(ref.extrinsic, query_camera.extrinsic)

        cameras.append(draw_camera(query_camera))

    o3d.visualization.draw_geometries([*cameras, o3d.geometry.TriangleMesh.create_coordinate_frame()])

