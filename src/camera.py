from dataclasses import dataclass, field, fields
from typing import Optional, Union, Dict

import numpy as np


@dataclass
class CameraExtrinsic:
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    T: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def E(self):
        E = np.eye(4)
        E[:3, -1] = self.T
        E[:3, :3] = self.R
        return E

    @property
    def C(self):
        return -self.R.T @ self.T


@dataclass
class CameraIntrinsic:
    K: np.ndarray = field(default_factory=lambda: np.eye(3))
    distortion_coefficients: np.ndarray = field(default_factory=lambda: np.zeros(5))

    def __post_init__(self):
        self.K = np.asarray(self.K)
        self.distortion_coefficients = np.asarray(self.distortion_coefficients)


@dataclass
class CameraMetadata:
    w: int
    h: int
    lat: float = None
    lng: float = None
    alt: float = None
    azn: float = None
    fov: float = None
    K: np.ndarray = None
    E: np.ndarray = None
    distortion_coefficients: np.ndarray = None

    @classmethod
    def from_kwargs(cls, **kwargs):
        # fetch the constructor's signature
        cls_fields = {field.name for field in fields(cls)}

        # split the kwargs into native ones and new ones
        native_args, new_args = {}, {}
        for name, val in kwargs.items():
            if name in cls_fields:
                native_args[name] = val
            else:
                new_args[name] = val

        # use the native ones to create the class ...
        ret = cls(**native_args)

        # ... and add the new ones by hand
        for new_name, new_val in new_args.items():
            setattr(ret, new_name, new_val)
        return ret


@dataclass
class Camera:
    intrinsic: CameraIntrinsic
    extrinsic: CameraExtrinsic
    metadata: Optional[CameraMetadata] = None

    @property
    def P(self):
        return np.concatenate((np.dot(self.intrinsic.K, self.extrinsic.R),
                               np.dot(self.intrinsic.K, self.extrinsic.T).reshape(-1, 1)), axis=1)
