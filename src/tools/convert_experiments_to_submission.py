import os.path

import numpy as np
from scipy.spatial.transform import Rotation

INPUT_DIR = 'test/aachen_expansion_policy_rerank'


def convert_transformation(E: np.ndarray):
    rx, ry, rz, rw = Rotation.from_matrix(E[:3, :3]).as_quat()
    t = E[:3, -1]
    return np.append(np.asarray([rw, rx, ry, rz]), t)


if __name__ == '__main__':
    filenames = np.load(os.path.join(INPUT_DIR, 'entries.npy'))
    poses = np.load(os.path.join(INPUT_DIR, 'estimated.npy'))

    lines = []
    for filename, pose in zip(filenames, poses):
        line = f'{filename}.jpg {" ".join(map(str, convert_transformation(pose)))}'
        lines.append(line)

    with open(f'{INPUT_DIR}.txt', 'w') as f:
        f.write('\n'.join(lines))
