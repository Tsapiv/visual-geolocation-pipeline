import os
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from geonavpy.retrieval.global_descriptor import GlobalDescriptor, generate_global_descriptors

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
    parser.add_argument('--descriptor_type', type=str, default='radenovic_gldv1')
    parser.add_argument('--backbone', type=str, default='resnet101conv5')
    parser.add_argument('--device', type=str, default='cuda')

    opt = parser.parse_args()

    device = torch.device(opt.device)

    model = GlobalDescriptor(model_type=opt.descriptor_type, backbone=opt.backbone)

    model = torch.nn.DataParallel(model)
    image_files = list(map(lambda x: os.path.join(x.path, 'image.jpg'), os.scandir(opt.dataset)))

    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            img = Image.open(img_path).convert('RGB')
            out = generate_global_descriptors(model, img, device)
            np.save(img_path.replace('image.jpg', f'{opt.descriptor_type}_descriptor.npy'), out.cpu().numpy())
