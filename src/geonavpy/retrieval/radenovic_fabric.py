from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.model_zoo import load_url

from ..retrieval.model import network

RADENOVIC = {
    'resnet50conv5_sfm': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'resnet101conv5_sfm': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'resnet50conv5_gldv1': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'resnet101conv5_gldv1': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth', }

WEIGHTS = Path(__file__).parent / 'weights'


def load(args: Namespace) -> torch.nn.Module:
    model = network.GeoLocalizationNet(args)

    url = RADENOVIC[f"{args.backbone}_{args.subtype}"]
    state_dict = load_url(url, model_dir=str(WEIGHTS))["state_dict"]
    model_keys = model.state_dict().keys()
    renamed_state_dict = {k: v for k, v in zip(model_keys, state_dict.values())}
    model.load_state_dict(renamed_state_dict)

    return model
