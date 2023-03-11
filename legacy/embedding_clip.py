import json
import os.path
from argparse import ArgumentParser

import clip
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils import match_paths, natural_ordering

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', required=True, type=str, choices=clip.available_models(), help='Name of the model')
    parser.add_argument('--input', required=True, type=str, help='Path to images folder')
    parser.add_argument('--output', required=False, type=str, default='./', help='Output folder')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    files = sorted(match_paths(args.input, r'.+\.(png|jpg|jpeg)'), key=natural_ordering)
    images = [Image.open(file) for file in files]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model, device=device)

    images = [preprocess(image).unsqueeze(0).to(device) for image in images]

    with torch.no_grad():
        features = [model.encode_image(image) for image in tqdm(images)]
        features = np.concatenate([np.squeeze(feature_batch.cpu().numpy()).reshape((1, -1)) for feature_batch in features])

    identifier = args.model.replace('/', '-')
    np.save(os.path.join(args.output, f'{identifier}-features.npy'), features)
    json.dump(files, open(os.path.join(args.output, f'{identifier}-id.json'), 'w'), indent=4)