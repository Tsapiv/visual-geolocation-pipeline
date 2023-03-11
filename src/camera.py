from dataclasses import dataclass, field
from typing import Optional

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


@dataclass
class Camera:
    intrinsic: CameraIntrinsic
    extrinsic: CameraExtrinsic
    metadata: Optional[CameraMetadata] = None

    @property
    def P(self):
        return np.concatenate((np.dot(self.intrinsic.K, self.extrinsic.R),
                               np.dot(self.intrinsic.K, self.extrinsic.T).reshape(-1, 1)), axis=1)
