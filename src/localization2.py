import datetime
import os

import cv2
import numpy as np
from tqdm import tqdm

from dataset import Dataset
from geoutils import camera_from_metadata
from keypoints_generation import generate_keypoints, tensor2numpy
from keypoints_matching import match_keypoints
from matching.superglue import SuperGlue
from matching.superpoint import SuperPoint
from pose_estimation import calculate_pose
from ranking.index import Index
from utils import compare_images, make_matching_plot

if __name__ == '__main__':

    verbose = True
    k = 10
    confidence_thr = 0.1
    distance_thr = 100
    evaluate = True

    # trainset = Dataset(root='datasets/aachen_v1_train', descriptor_type='radenovic_gldv1')
    # testset = Dataset(root='datasets/aachen_v1_nighttime', descriptor_type='radenovic_gldv1')

    dataset = Dataset(root='datasets/Lviv49.8443931@24.0254815', descriptor_type='radenovic_gldv1')

    # train_entries, test_entries = train_test_split(dataset.entries, test_size=0.2)
    err = np.load('test/lviv0/err.npy')
    entries = np.load('test/lviv0/entries.npy')
    test_entries = set(entries[err > 15])
    train_entries = set(dataset.entries) - set(entries)

    trainset = dataset.get_subset(train_entries)
    testset = dataset.get_subset(test_entries)

    super_point = SuperPoint()
    super_glue = SuperGlue(dict(weights='outdoor'))

    train_index = Index(trainset)

    err = []
    gt_poses = []
    estimated_poses = []
    entries = []
    retrieved_entries = []

    for entry in tqdm(testset.entries):
        try:
            query_image = testset.image(entry)
            query_camera = camera_from_metadata(testset.metadata(entry))

            # skip step with descriptor calculation
            query_descriptor = testset.descriptor(entry)

            similar_entries = train_index.topk(query_descriptor, k, False)
            keypoints = trainset.keypoint(similar_entries)

            if verbose:
                compare_images(query_image, trainset.image(similar_entries))

            query_keypoint = generate_keypoints(super_point, query_image)

            refined_query_keypoint = None
            refined_keypoints = []
            matches = []
            match_confidences = []

            for i, keypoint in enumerate(keypoints):
                pred = match_keypoints(super_glue, query_keypoint, keypoint)

                refined_query_keypoint = np.squeeze(tensor2numpy(pred['keypoints0']))
                refined_keypoints.append(np.squeeze(tensor2numpy(pred['keypoints1'])))
                matches.append(np.squeeze(tensor2numpy(pred['matches'])))
                match_confidences.append(np.squeeze(tensor2numpy(pred['match_confidence'])))

                if verbose:
                    make_matching_plot(query_image,
                                       trainset.image(similar_entries[i]),
                                       refined_query_keypoint,
                                       refined_keypoints[i],
                                       matches[i],
                                       match_confidences[i],
                                       show_keypoints=True,
                                       opencv_display=True)

            cameras = [camera_from_metadata(m) for m in trainset.metadata(similar_entries, cache=True)]

            ret = calculate_pose(matches,
                                 match_confidences,
                                 refined_keypoints,
                                 cameras,
                                 refined_query_keypoint,
                                 query_camera,
                                 confidence_thr=confidence_thr,
                                 distance_thr=distance_thr,
                                 verbose=verbose)
            if ret is None:
                continue
            estimated_camera, base_camera = ret

            C0 = estimated_camera.extrinsic.C
            if evaluate:
                C1 = query_camera.extrinsic.C
                print(f'Estimated C: {C0}')
                print(f'GT C: {C1}')
                print(f'Err: {np.linalg.norm(C0 - C1)}m')
                err.append(np.linalg.norm(C0 - C1))
                gt_poses.append(query_camera.extrinsic)
                estimated_poses.append(estimated_camera.extrinsic)
                retrieved_entries.append(similar_entries)
                entries.append(entry)
            else:
                C1 = base_camera.extrinsic.C
                print(f'Estimated C: {C0}')
                print(f'Base camera C: {C1}')
                print(f'Distance: {np.linalg.norm(C0 - C1)}m')
        except Exception as e:
            print(e)
            raise

    if evaluate:
        test_dir = f'test/{datetime.datetime.now().isoformat()}'
        os.makedirs(test_dir, exist_ok=True)
        err = np.asarray(err)
        gt_poses = np.asarray(gt_poses)
        estimated_poses = np.asarray(estimated_poses)
        entries = np.asarray(entries)
        retrieved_entries = np.asarray(retrieved_entries)
        np.save(os.path.join(test_dir, 'err.npy'), err)
        np.save(os.path.join(test_dir, 'gt.npy'), gt_poses)
        np.save(os.path.join(test_dir, 'estimated.npy'), estimated_poses)
        np.save(os.path.join(test_dir, 'retrieval.npy'), retrieved_entries)
        np.save(os.path.join(test_dir, 'entries.npy'), entries)
        print(np.median(err))
        print(err.mean())
        print(err.std())
        print(np.max(err))
        print(np.min(err))
