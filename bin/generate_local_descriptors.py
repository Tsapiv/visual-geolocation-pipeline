import argparse
import os

import numpy as np
import tqdm

from geonavpy.keypoints_generation import generate_keypoints, tensor2numpy
from geonavpy.matching.superpoint import SuperPoint
from geonavpy.utils import match_paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--database', type=str, required=True,
        help='Path to root directory of the dataset')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
             ' (Must be positive)')
    opt = vars(parser.parse_args())

    model = SuperPoint(opt)

    for image_path in tqdm.tqdm(match_paths(opt['database'], r'.+\.(png|jpg|jpeg)')):
        dirname = os.path.dirname(image_path)
        keypoints = generate_keypoints(model, image_path)
        keypoints = {key: tensor2numpy(value) for key, value in keypoints.items()}
        np.savez(os.path.join(dirname, 'keypoints.npz'), **keypoints)