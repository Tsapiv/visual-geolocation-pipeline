from typing import Union, Optional, List

import numpy as np
import open3d as o3d

from ..common.camera import Camera


def draw_camera(camera: Camera):
    geometry = o3d.geometry.LineSet().create_camera_visualization(int(camera.metadata.w), int(camera.metadata.h),
                                                                  camera.intrinsic.K,
                                                                  camera.extrinsic.E)
    return geometry


def visualize_reconstruction(pcd: Union[np.ndarray, o3d.geometry.PointCloud], cameras: List[Camera], gt_camera: Camera,
                             estimated_camera: Optional[Camera]):
    if isinstance(pcd, np.ndarray):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    for camera in cameras:
        if np.allclose(camera.extrinsic.C, gt_camera.extrinsic.C):
            continue
        vis.add_geometry(draw_camera(camera))

    if gt_camera.extrinsic is not None:
        gt_camera_geom = draw_camera(gt_camera)
        gt_camera_geom.paint_uniform_color((0, 1, 0))
        vis.add_geometry(gt_camera_geom)

    if estimated_camera is not None:
        estimated_camera_geom = draw_camera(estimated_camera)
        estimated_camera_geom.paint_uniform_color((1, 0, 0))
        vis.add_geometry(estimated_camera_geom)

        ctr: o3d.visualization.ViewControl = vis.get_view_control()
        ctr.change_field_of_view(step=90)
        par: o3d.camera.PinholeCameraParameters = ctr.convert_to_pinhole_camera_parameters()
        par.extrinsic = estimated_camera.extrinsic.E
        ctr.convert_from_pinhole_camera_parameters(par)

    vis.run()
    vis.destroy_window()
