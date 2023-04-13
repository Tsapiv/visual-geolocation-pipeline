import json
import os
from typing import Dict, Optional, Iterable, Union

import cv2
import numpy as np


class Dataset:

    def __init__(self, root: str, descriptor_type: Optional[str] = None):
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

    @property
    def root(self):
        return self.__root

    def __get_descriptor(self, key: str, cache: bool):
        if key in self.__descriptors:
            return self.__descriptors[key]
        if self.__descriptor_type is None:
            raise ValueError('Missing descriptor type')
        descriptor = np.load(os.path.join(self.__root, key, f'{self.__descriptor_type}_descriptor.npy'))
        if cache:
            self.__descriptors[key] = descriptor
        return descriptor

    def __get_image(self, key: str, cache: bool):
        if key in self.__images:
            return self.__images[key]

        image = cv2.imread(os.path.join(self.__root, key, 'image.jpg'))
        if cache:
            self.__images[key] = image
        return image

    def __get_keypoint(self, key: str, cache: bool):
        if key in self.__keypoints:
            return self.__keypoints[key]

        npzfile = np.load(os.path.join(self.__root, key, 'keypoints.npz'))
        keypoint = {name: npzfile[name] for name in npzfile.files}
        if cache:
            self.__keypoints[key] = keypoint
        return keypoint

    def __get_metadata(self, key: str, cache: bool):
        if key in self.__metadata:
            return self.__metadata[key]

        with open(os.path.join(self.__root, key, 'metadata.json')) as f:
            metadata = json.load(f)
        if cache:
            self.__metadata[key] = metadata
        return metadata

    def image(self, key: Union[str, Iterable[str]], cache: bool = False):
        if isinstance(key, str):
            return self.__get_image(key, cache)
        else:
            return [self.__get_image(e, cache) for e in key]

    def descriptor(self, key: Union[str, Iterable[str]], cache: bool = False):
        if isinstance(key, str):
            return self.__get_descriptor(key, cache)
        else:
            return [self.__get_descriptor(e, cache) for e in key]

    def keypoint(self, key: Union[str, Iterable[str]], cache: bool = False):
        if isinstance(key, str):
            return self.__get_keypoint(key, cache)
        else:
            return [self.__get_keypoint(e, cache) for e in key]

    def metadata(self, key: Union[str, Iterable[str]], cache: bool = False):
        if isinstance(key, str):
            return self.__get_metadata(key, cache)
        else:
            return [self.__get_metadata(e, cache) for e in key]

    def get_subset(self, entries: Iterable[str]):
        subset = self.__class__(self.__root, self.__descriptor_type)
        subset.__entries = list(entries)
        return subset
