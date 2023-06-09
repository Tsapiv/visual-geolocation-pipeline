from argparse import Namespace
from copy import deepcopy
from typing import Optional, Literal, Union

import numpy as np
import torch
from PIL.Image import Image
from torch import nn
from torchvision import transforms as T

from .naver_fabric import load as load_naver
from .radenovic_fabric import load as load_radenovic

default_config = {'aggregation': 'gem', 'fc_output_dim': 2048, 'backbone': 'resnet101conv5', 'pretrain': True,
                  'type': 'radenovic', 'subtype': 'gldv1', 'l2': 'after_pool'}

transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class GlobalDescriptor(nn.Module):

    def __init__(self, model_type: Optional[Literal['naver', 'radenovic_gldv1', 'radenovic_sfm']] = None,
                 backbone: Optional[Literal['resnet50conv5', 'resnet101conv5']] = None):
        super().__init__()
        params = deepcopy(default_config)
        if model_type is not None and model_type.startswith('naver'):
            params.update(l2='none')
            params.update(type=model_type)
            params.update(subtype='')
        elif model_type is not None and model_type.startswith('radenovic'):
            params.update(l2='after_pool')
            params.update(type=model_type.split('_')[0])
            params.update(subtype=model_type.split('_')[-1])

        if backbone is not None:
            params.update(backbone=backbone)

        if params['type'] == 'naver':
            self.model = load_naver(Namespace(**params))
        elif params['type'] == 'radenovic':
            self.model = load_radenovic(Namespace(**params))
        else:
            raise ValueError(f'Unknown model type "{params["type"]}"')

    def forward(self, x):
        return self.model(x)


def generate_global_descriptors(model: GlobalDescriptor, image: Union[np.ndarray, Image],
                                device: Optional[torch.device] = None):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)
    model.eval()

    image = transform(image)
    if len(image.shape) == 3:
        image = torch.unsqueeze(image, 0)
    image = image.to(device)

    with torch.no_grad():
        descriptor = torch.squeeze(model(image))
        if len(descriptor.shape) == 1:
            descriptor = torch.unsqueeze(descriptor, 0)
        return descriptor
