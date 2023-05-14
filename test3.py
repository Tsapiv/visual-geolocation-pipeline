import numpy as np

from dataset import Dataset

if __name__ == '__main__':
    PATH = 'datasets/aachen_v1_test'
    d = Dataset(PATH)
    np.save('aachen_v1_test.npy', np.asarray(d.entries))