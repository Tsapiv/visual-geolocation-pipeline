import json
import os
from typing import Dict, Optional, Iterable, Union

import cv2
import numpy as np


class Dataset:

    def __init__(self, root: str, descriptor_type: str):
        self.__root = root
        self.__descriptor_type = descriptor_type
        self.__entries = [e.name for e in os.scandir(self.__root) if e.is_dir()]
        self.__descriptors: Optional[Dict[str, np.ndarray]] = {}
        self.__keypoints: Optional[Dict[str, Dict[str, np.ndarray]]] = {}
        self.__images: Optional[Dict[str, np.ndarray]] = {}
        self.__metadata: Optional[Dict[str, Dict]] = {}

    @property
    def entries(self):
        return self.__entries

    def __get_cached_descriptor(self, key: str):
        if key in self.__descriptors:
            return self.__descriptors[key]
        files = list(filter(lambda x: x.endswith('descriptor.npy'), os.listdir(os.path.join(self.__root, key))))
        assert len(files) == 1
        descriptor = np.load(os.path.join(self.__root, key, f'{self.__descriptor_type}_descriptor.npy'))
        self.__descriptors[key] = descriptor
        return descriptor

    def __get_cached_image(self, key: str):
        if key in self.__images:
            return self.__images[key]

        image = cv2.imread(os.path.join(self.__root, key, 'image.jpg'))
        self.__images[key] = image
        return image

    def __get_cached_keypoint(self, key: str):
        if key in self.__keypoints:
            return self.__keypoints[key]

        npzfile = np.load(os.path.join(self.__root, key, 'keypoints.npz'))
        keypoint = {name: npzfile[name] for name in npzfile.files}
        self.__keypoints[key] = keypoint
        return keypoint

    def __get_cached_metadata(self, key: str):
        if key in self.__metadata:
            return self.__metadata[key]

        with open(os.path.join(self.__root, key, 'metadata.json')) as f:
            metadata = json.load(f)
        self.__metadata[key] = metadata
        return metadata

    def image(self, key: Union[str, Iterable[str]]):
        if isinstance(key, str):
            return self.__get_cached_image(key)
        else:
            return [self.__get_cached_image(e) for e in key]

    def descriptor(self, key: Union[str, Iterable[str]]):
        if isinstance(key, str):
            return self.__get_cached_descriptor(key)
        else:
            return [self.__get_cached_descriptor(e) for e in key]

    def keypoint(self, key: Union[str, Iterable[str]]):
        if isinstance(key, str):
            return self.__get_cached_keypoint(key)
        else:
            return [self.__get_cached_keypoint(e) for e in key]

    def metadata(self, key: Union[str, Iterable[str]]):
        if isinstance(key, str):
            return self.__get_cached_metadata(key)
        else:
            return [self.__get_cached_metadata(e) for e in key]
