from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .pose_estimation import estimate_pose, RP
from ..common.camera import Camera
from ..common.dataset import Dataset
from ..geometric_validation import filter_distant_cameras
from ..geoutils import camera_from_metadata
from ..keypoints_generation import generate_keypoints, tensor2numpy
from ..keypoints_matching import match_keypoints
from ..matching.superglue import SuperGlue
from ..matching.superpoint import SuperPoint
from ..ranking.index import Index
from ..utils import make_retrieval_plot, make_matching_plot


@dataclass
class Params:
    k: int
    confidence_thr: float
    points_distance_thr: float
    cameras_distance_thr: float
    rerank: bool
    policy: RP
    verbose: bool = False

    def __post_init__(self):
        if isinstance(self.policy, str):
            self.policy = RP[self.policy]


class Localizer:

    def __init__(self, config: Dict, reference_db: Dataset):
        self.__config = Params(**config)
        self.__reference_db = reference_db
        self.__local_descriptor = SuperPoint()
        self.__matcher = SuperGlue(dict(weights='outdoor'))
        self.__index_db = Index(reference_db)
        self.__global_descriptor = None

    def localize(self, query_image: np.ndarray, query_camera: Camera, query_descriptor: Optional[np.ndarray] = None):
        if query_descriptor is None and self.__global_descriptor is not None:
            query_descriptor = self.__global_descriptor(query_image)

        similar_entries = self.__index_db.topk(query_descriptor, self.__config.k, self.__config.rerank)
        cameras = np.asarray(
            [camera_from_metadata(m) for m in self.__reference_db.metadata(similar_entries, cache=True)])
        valid = filter_distant_cameras(cameras, self.__config.cameras_distance_thr)

        if valid.size < 2:
            self.__config.verbose and print('Retrieved image are too far apart')
            return

        similar_entries = similar_entries[valid]
        cameras = cameras[valid]

        keypoints = self.__reference_db.keypoint(similar_entries)

        if self.__config.verbose:
            make_retrieval_plot(query_image, self.__reference_db.image(similar_entries))

        query_keypoint = generate_keypoints(self.__local_descriptor, query_image)

        refined_query_keypoint = None
        refined_keypoints = []
        matches = []
        match_confidences = []

        for i, keypoint in enumerate(keypoints):
            pred = match_keypoints(self.__matcher, query_keypoint, keypoint)

            refined_query_keypoint = np.squeeze(tensor2numpy(pred['keypoints0']))
            refined_keypoints.append(np.squeeze(tensor2numpy(pred['keypoints1'])))
            matches.append(np.squeeze(tensor2numpy(pred['matches'])))
            match_confidences.append(np.squeeze(tensor2numpy(pred['match_confidence'])))

            if self.__config.verbose:
                make_matching_plot(query_image, self.__reference_db.image(similar_entries[i]), refined_query_keypoint,
                                   refined_keypoints[i], matches[i], match_confidences[i], show_keypoints=False,
                                   opencv_display=True)

        pose = estimate_pose(matches, match_confidences, refined_keypoints, cameras, refined_query_keypoint,
                             query_camera, policy=self.__config.policy, confidence_thr=self.__config.confidence_thr,
                             distance_thr=self.__config.points_distance_thr, verbose=self.__config.verbose)
        return pose
