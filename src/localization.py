import random
from typing import Dict

import numpy as np
from tqdm import tqdm

from camera import Camera, CameraExtrinsic, CameraIntrinsic, CameraMetadata
from dataset import Dataset
from pose_estimation import calculate_pose, get_relative_pose
from ranking.index import Index
from matching.superpoint import SuperPoint
from matching.superglue import SuperGlue

from keypoints_generation import generate_keypoints, tensor2numpy
from keypoints_matching import match_keypoints
from geoutils import relative_extrinsic_from_metadata, intrinsics_from_metadata


if __name__ == '__main__':

    trainset = Dataset(root='datasets/Lviv49.8443931@24.0254815', descriptor_type='radenovic_gldv1')
    testset = Dataset(root='datasets/Lviv49.8443931@24.0254815', descriptor_type='radenovic_gldv1')

    super_point = SuperPoint({})
    super_glue = SuperGlue({'weights': 'outdoor'})

    train_index = Index(trainset)

    for entry in tqdm(random.choices(testset.entries, k=50)):
        query_image = testset.image(entry)
        query_metadata = CameraMetadata(**testset.metadata(entry))

        # skip step with descriptor calculation
        query_descriptor = testset.descriptor(entry)

        best_entries = train_index.topk(query_descriptor, 6, True)[1:]  # skip the first because same dataset is used
        keypoints = trainset.keypoint(best_entries)

        metadata = [CameraMetadata(**m) for m in trainset.metadata(best_entries)]

        cameras = []
        for m in metadata:
            cameras.append(Camera(intrinsics_from_metadata(m), relative_extrinsic_from_metadata(metadata[0], m)))

        query_camera = Camera(intrinsics_from_metadata(query_metadata),
                              relative_extrinsic_from_metadata(metadata[0], query_metadata))


        query_keypoint = generate_keypoints(super_point, query_image)

        refined_query_keypoint = None
        refined_keypoints = []
        matches = []
        match_confidences = []

        for keypoint in keypoints:
            pred = match_keypoints(super_glue, query_keypoint, keypoint)

            refined_query_keypoint = np.squeeze(tensor2numpy(pred['keypoints0']))
            refined_keypoints.append(np.squeeze(tensor2numpy(pred['keypoints1'])))
            matches.append(np.squeeze(tensor2numpy(pred['matches'])))
            match_confidences.append(np.squeeze(tensor2numpy(pred['match_confidence'])))

        ret = calculate_pose(matches,
                             match_confidences,
                             refined_keypoints,
                             cameras,
                             refined_query_keypoint,
                             query_camera,
                             confidence_thr=0.2,
                             verbose=False)
        if ret is None:
            continue
        estimated_camera, base_camera = ret

        C0 = estimated_camera.extrinsic.C
        C1 = get_relative_pose(base_camera.extrinsic, query_camera.extrinsic).C
        print(f'C0: {C0}')
        print(f'C1: {C1}')
        print(f'Err: {np.linalg.norm((C0 - C1)[::2])}m')
