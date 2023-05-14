import argparse
import os
from typing import Union, Optional

import cv2
import numpy as np
import torch
import tqdm

from .matching.superpoint import SuperPoint
from .matching.utils import frame2tensor
from .utils import match_paths

torch.set_grad_enabled(False)


def tensor2numpy(tensor: Union[torch.Tensor, np.ndarray]):
    if isinstance(tensor, np.ndarray):
        return tensor
    else:
        return tensor.cpu().numpy()


def generate_keypoints(model: SuperPoint, image: Union[str, np.ndarray], device: Optional[torch.device] = None):
    if isinstance(image, str):
        inp = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if inp is None:
            raise FileNotFoundError(image)
    else:
        if len(image.shape) == 3:
            inp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            inp = image
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        pred = model(dict(image=frame2tensor(inp, device)))
        return {**pred, **{k: torch.stack(v) for k, v in pred.items() if isinstance(v, tuple) or isinstance(v, list)},
                'image': inp}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    parser.add_argument(
        '--dataset', type=str, required=True,
        help='Path to root directory of the dataset')
    opt = vars(parser.parse_args())

    model = SuperPoint(opt)

    for image_path in tqdm.tqdm(match_paths(opt['dataset'], r'.+\.(png|jpg|jpeg)')):
        dirname = os.path.dirname(image_path)
        keypoints = generate_keypoints(model, image_path)
        keypoints = {key: tensor2numpy(value) for key, value in keypoints.items()}
        np.savez(os.path.join(dirname, 'keypoints.npz'), **keypoints)
