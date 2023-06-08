from pathlib import Path

import numpy as np

from geonavpy.common.camera import CameraExtrinsic
from geonavpy.matching.utils import angle_error_mat

EXPERIMENTS = [
    # Path('test/lviv_center_displacement_policy_no_rerank'),
    Path('test/lviv_center2_displacement_policy_no_rerank'),
    # Path('test/lviv_center_expansion_policy_no_rerank'),
    Path('test/lviv_center2_expansion_policy_no_rerank'),
    # Path('test/lviv_center_displacement_policy_rerank'),
    Path('test/lviv_center2_displacement_policy_rerank'),
    # Path('test/lviv_center_expansion_policy_rerank'),
    Path('test/lviv_center2_expansion_policy_rerank')
]

THRESHOLDS = [(0.5, 2), (1, 5), (5, 10)]


def accuracy(t_err, r_err, t_thr, r_thr):
    return round(np.mean((t_err <= t_thr) & (r_err <= r_thr)), 2)


if __name__ == '__main__':

    for experiment in EXPERIMENTS:
        gt = np.load(str(experiment / 'gt.npy'))
        estimated = np.load(str(experiment / 'estimated.npy'))

        t_err = []
        r_err = []
        for gt_E, estimated_E in zip(gt, estimated):
            gt_cam = CameraExtrinsic.from_E(gt_E)
            estimated_cam = CameraExtrinsic.from_E(estimated_E)

            t_err.append(np.linalg.norm(gt_cam.C - estimated_cam.C))
            r_err.append(angle_error_mat(gt_cam.R, estimated_cam.R))
        t_err = np.asarray(t_err)
        r_err = np.asarray(r_err)
        for thr in THRESHOLDS:
            print(accuracy(t_err, r_err, *thr), end='/')

        print(f'\n {len(t_err)}')

        t_err = t_err[~np.isnan(t_err)]
        print(experiment)
        print(f'{np.median(t_err) = }')
        print(f'{np.median(r_err) = }')
        print('=' * 20)
