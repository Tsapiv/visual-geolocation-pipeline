from typing import Union, Optional, Dict

import numpy as np
import torch

from matching.superglue import SuperGlue

torch.set_grad_enabled(False)


def to_tensor(arr: Union[np.ndarray, torch.Tensor]):
    if isinstance(arr, np.ndarray):
        return torch.squeeze(torch.from_numpy(arr).float())
    return torch.squeeze(arr)


def match_keypoints(model: SuperGlue, keypoints0: Dict[str, Union[np.ndarray, torch.Tensor]],
                    keypoints1: Dict[str, Union[np.ndarray, torch.Tensor]],
                    device: Optional[torch.device] = None):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)
    model.eval()

    keypoints0 = {k + '0': torch.unsqueeze(to_tensor(v).to(device), 0) for k, v in keypoints0.items()}
    keypoints1 = {k + '1': torch.unsqueeze(to_tensor(v).to(device), 0) for k, v in keypoints1.items()}

    with torch.no_grad():
        pred = model({**keypoints0, **keypoints1})

        return {'keypoints0': keypoints0['keypoints0'], 'keypoints1': keypoints1['keypoints1'],
                'matches': pred['matches0'], 'match_confidence': pred['matching_scores0']}
