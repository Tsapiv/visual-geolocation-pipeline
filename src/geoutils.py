import numpy as np
from pyproj import Geod
from scipy.spatial.transform import Rotation

from camera import CameraMetadata, CameraIntrinsic, CameraExtrinsic

GEOD = Geod(ellps='WGS84')


def intrinsics_from_metadata(metadata: CameraMetadata):
    if metadata.K is not None:
        return CameraIntrinsic(K=metadata.K)
    K = np.array([[metadata.w / np.tan(np.deg2rad(metadata.fov / 2)) / 2, 0, metadata.w / 2],
                  [0, metadata.h / np.tan(np.deg2rad(metadata.fov / 2)) / 2, metadata.h / 2],
                  [0, 0, 1]])
    return CameraIntrinsic(K=K)


def relative_camera_position_from_metadata(metadata1: CameraMetadata, metadata2: CameraMetadata):
    azimuths_deg, _, dist = GEOD.inv(metadata1.lng, metadata1.lat, metadata2.lng, metadata2.lat)

    angel = np.deg2rad(90 - (azimuths_deg - metadata1.azn))

    C12 = np.array([np.cos(angel) * dist, -(metadata2.alt - metadata1.alt), np.sin(angel) * dist])
    return C12


def relative_extrinsic_from_metadata(metadata1: CameraMetadata, metadata2: CameraMetadata):
    C12 = relative_camera_position_from_metadata(metadata1, metadata2)

    R12 = Rotation.from_rotvec(np.array([0, np.deg2rad(-(metadata2.azn - metadata1.azn)), 0])).as_matrix()

    T12 = -R12 @ C12

    return CameraExtrinsic(R=R12, T=T12)
