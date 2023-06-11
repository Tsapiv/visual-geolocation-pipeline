from typing import Union, Optional

import cv2
import numpy as np
import torch

from .matching.superpoint import SuperPoint
from .matching.utils import frame2tensor

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
