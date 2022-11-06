import numpy as np
import json
import cv2

if __name__ == '__main__':
    features = np.squeeze(np.load('data/region_filtered_ViT-L14_features.npy'))
    filenames = json.load(open('data/region_filtered_ViT-L14_filenames.json'))
    n = 10

    # distance = np.squeeze(np.sqrt(np.sum(features - features[0], axis=-1) ** 2))
    distance = features @ features[n] / (np.linalg.norm(features, axis=-1) * np.linalg.norm(features[n]))


    indexes = np.argsort(distance)[::-1]
    anchor = cv2.imread(filenames[indexes[0]])

    for idx in indexes:
        window = f'{filenames[indexes[0]]}\t\t{filenames[idx]} -- similarity: {distance[idx]}'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        image = cv2.imread(filenames[idx])
        cv2.imshow(window, np.concatenate((anchor, image), axis=1))
        cv2.waitKey()
        cv2.destroyAllWindows()
        print(f'File: {filenames[idx]}, distance: {distance[idx]}')
