import datetime
import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from camera import Camera, CameraMetadata
from dataset import Dataset
from geoutils import intrinsics_from_metadata, extrinsic_from_metadata
from keypoints_generation import generate_keypoints, tensor2numpy
from keypoints_matching import match_keypoints
from matching.superglue import SuperGlue
from matching.superpoint import SuperPoint
from pose_estimation import calculate_pose
from ranking.index import Index


if __name__ == '__main__':

    verbose = False
    k = 10
    confidence_thr = 0.3
    distance_thr = 100
    evaluate = True

    # trainset = Dataset(root='datasets/aachen_v1_train', descriptor_type='radenovic_gldv1')
    # testset = Dataset(root='datasets/aachen_v1_nighttime', descriptor_type='radenovic_gldv1')

    dataset = Dataset(root='datasets/aachen_v1_train', descriptor_type='radenovic_gldv1')

    train_entries, test_entries = train_test_split(dataset.entries, test_size=0.25)

    trainset = dataset.get_subset(train_entries)
    testset = dataset.get_subset(test_entries)


    super_point = SuperPoint({})
    super_glue = SuperGlue({'weights': 'outdoor'})

    train_index = Index(trainset)

    err = []
    gt_poses = []
    estimated_poses = []

    for entry in tqdm(testset.entries):
        try:
            query_image = testset.image(entry)
            query_metadata = CameraMetadata(**testset.metadata(entry))

            # skip step with descriptor calculation
            query_descriptor = testset.descriptor(entry)

            best_entries = train_index.topk(query_descriptor, k, True)
            keypoints = trainset.keypoint(best_entries)


            if verbose:
                images = list(trainset.image(best_entries))
                shapes = np.asarray(list(map(lambda x: x.shape[:2], images + [query_image])))
                max_h, max_w = np.max(shapes, axis=0)
                print(max_h, max_w)
                #
                anchor = query_image
                anchor_padded = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                anchor_padded[:anchor.shape[0], :anchor.shape[1], :] = anchor
                window = f'Compare'
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                for rank, image in enumerate(images):
                    image_padded = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                    image_padded[:image.shape[0], :image.shape[1], :] = image
                    cv2.imshow(window, np.concatenate((anchor_padded, image_padded), axis=1))
                    cv2.waitKey()
                cv2.destroyAllWindows()

            metadata = [CameraMetadata(**m) for m in trainset.metadata(best_entries, cache=True)]

            cameras = []
            for m in metadata:
                cameras.append(Camera(intrinsics_from_metadata(m), extrinsic_from_metadata(m), metadata=m))

            query_camera = Camera(intrinsics_from_metadata(query_metadata), extrinsic_from_metadata(query_metadata), metadata=query_metadata)


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
            else:
                C1 = base_camera.extrinsic.C
                print(f'Estimated C: {C0}')
                print(f'Base camera C: {C1}')
                print(f'Distance: {np.linalg.norm(C0 - C1)}m')
        except Exception as e:
            print(e)

    if evaluate:
        test_dir = f'test/{datetime.datetime.now().isoformat()}'
        os.makedirs(test_dir, exist_ok=True)
        err = np.asarray(err)
        gt_poses = np.asarray(gt_poses)
        estimated_poses = np.asarray(estimated_poses)
        np.save(os.path.join(test_dir, 'err.npy'), err)
        np.save(os.path.join(test_dir, 'gt.npy'), gt_poses)
        np.save(os.path.join(test_dir, 'estimated.npy'), estimated_poses)
        print(np.median(err))
        print(err.mean())
        print(err.std())
        print(np.max(err))
        print(np.min(err))

