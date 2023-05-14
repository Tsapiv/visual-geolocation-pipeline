import math

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
import cv2



if __name__ == '__main__':
    err = np.load('test/street_displacement_policy_no_rerank/err.npy')
    print(err.size)
    err = err[~np.isnan(err)]
    print(f'{np.median(err) = }')
    print(np.sum(err > 15))
    print(np.sum(np.isnan(err)))
    print(np.mean(err <= 0.25))
    print(np.mean(err <= 0.5))
    print(np.mean(err <= 1))
    print(np.mean(err <= 5))
    print(np.mean(err <= 13))
    err = err.clip(max=10)
    plt.hist(err)
    plt.show()