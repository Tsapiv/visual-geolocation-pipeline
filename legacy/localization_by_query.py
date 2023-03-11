from argparse import ArgumentParser
from typing import List

import numpy as np
import json
import cv2
import torch
from clip import clip
from sklearn.metrics.pairwise import cosine_distances


def extract_model_name(name: str):
    name = name.split('/')[-1]
    parts = name.split('-')
    if len(parts) != 3:
        return name
    else:
        return ''.join(['-'.join(parts[:-1]), '/', parts[-1]])


if __name__ == '__main__':
    print(clip.available_models())
    parser = ArgumentParser()
    parser.add_argument('--input', required=True, type=str, help='Prefix to *_features.npy and *_id.json')
    parser.add_argument('-k', required=False, type=int, default=3)
    parser.add_argument('-v', action='store_true', help='Option for visualization')

    args = parser.parse_args()

    features = np.squeeze(np.load(f'{args.input}-features.npy'))
    ids: List[str] = json.load(open(f'{args.input}-id.json'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = extract_model_name(args.input)
    model, _ = clip.load(model_name, device=device)

    while True:
        with torch.no_grad():
            query = input('Query: ')
            text = clip.tokenize(query).to(device)
            feature = model.encode_text(text)

        feature = feature.cpu().numpy()

        identifiers = np.asarray(ids)
        distance = np.squeeze(cosine_distances(features, feature))


        order = np.argsort(distance)

        identifiers = identifiers[order]
        window = f'Compare'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        for filename in identifiers[:5]:
            image = cv2.imread(filename.item())
            cv2.imshow(window, image)
            cv2.waitKey()
        cv2.destroyAllWindows()
