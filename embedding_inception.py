import json
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models.inception import Inception3, Inception_V3_Weights
from tqdm import tqdm

from utils import natural_ordering, match_paths


class CustomInception3(Inception3):
    IMAGE_SIZE = 299
    preprocess = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         transforms.Resize(IMAGE_SIZE)])

    def __init__(self, final_pooling=None):
        self.final_pooling = final_pooling
        super(CustomInception3, self).__init__()

    def forward(self, x):
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)  # size (batch_size, 2048, 1, 1)
        # 1 x 1 x 2048

        if self.final_pooling:
            x = F.avg_pool1d(x.view(x.size(0), 2048, 1).permute(0, 2, 1), kernel_size=self.final_pooling)

        return x


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', required=False, type=str, default='Inception3', choices=['Inception3'],
                        help='Name of the model')
    parser.add_argument('--input', required=True, type=str, help='Path to images folder')
    parser.add_argument('--output', required=False, type=str, default='./', help='Output folder')
    parser.add_argument('--chunk-size', required=False, type=int, default=100, help='Size of chunk to process')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CustomInception3(final_pooling=8)
    model.load_state_dict(Inception_V3_Weights.DEFAULT.get_state_dict(progress=True))
    model.to(device)
    model.eval()
    print('Model is loaded')

    files = sorted(match_paths(args.input, r'.+\.(png|jpg|jpeg)'), key=natural_ordering)
    file_chunks = [files[i:i + args.chunk_size] for i in range(0, len(files), args.chunk_size)]

    features = []
    with torch.no_grad():
        for chunk in tqdm(file_chunks):
            images = [Image.open(file) for file in chunk]
            images = [model.preprocess(image) for image in images]
            batch = torch.stack(images).to(device)
            del images
            feature_batch = model(batch)
            features.append(np.squeeze(feature_batch.cpu().numpy()))
        features = np.concatenate(features)
        identifier = args.model.replace('/', '-')
        np.save(os.path.join(args.output, f'{identifier}-features.npy'), features)
        json.dump(files, open(os.path.join(args.output, f'{identifier}-id.json'), 'w'), indent=4)
