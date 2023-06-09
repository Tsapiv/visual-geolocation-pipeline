import sys
from argparse import Namespace
from pathlib import Path

import sklearn.decomposition
import torch

from ..retrieval.model import network

NAVER = {
    'resnet50conv5': 'Resnet50-AP-GeM.pt',
    'resnet101conv5': 'Resnet-101-AP-GeM.pt'
}

WEIGHTS = Path(__file__).parent / 'weights'


def load(args: Namespace) -> torch.nn.Module:
    model = network.GeoLocalizationNet(args)
    # model = model.to(args.device)

    sys.modules['sklearn.decomposition.pca'] = sklearn.decomposition._pca

    state_dict_path = WEIGHTS / NAVER[args.backbone]
    if not state_dict_path.exists():
        raise FileNotFoundError(state_dict_path)
    state_dict = torch.load(state_dict_path)
    state_dict = state_dict["state_dict"]
    model_keys = model.state_dict().keys()
    renamed_state_dict = {k: v for k, v in zip(model_keys, state_dict.values())}
    model.load_state_dict(renamed_state_dict)

    return model
