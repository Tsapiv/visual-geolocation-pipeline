import datetime
import os
import traceback
from argparse import ArgumentParser

import numpy as np
import yaml
from tqdm import tqdm

from geonavpy.common.dataset import Dataset
from geonavpy.geoutils import camera_from_metadata
from geonavpy.localization.localizer import Localizer

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--reference_set', type=str, required=True, help='Path to reference dataset')
    parser.add_argument('--descriptor_type', type=str, default='radenovic_gldv1', help='Type of global descriptor used in retrieval')
    parser.add_argument('--query_set', type=str, required=True, help='Path to query dataset')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--conf', type=str, required=True, help='Path to config file')

    opt = parser.parse_args()

    conf = yaml.safe_load(open(opt.conf))

    reference_set = Dataset(root=opt.reference_set, descriptor_type=opt.descriptor_type)
    query_set = Dataset(root=opt.reference_set, descriptor_type=opt.descriptor_type)

    localizer = Localizer(conf, reference_set)

    gt = []
    estm = []
    entries = []

    for entry in tqdm(query_set.entries):
        try:
            query_image = query_set.image(entry)
            query_camera = camera_from_metadata(query_set.metadata(entry))

            estimated_camera = localizer.localize(query_image, query_camera)

            if query_camera.extrinsic is None:
                gt.append(np.zeros(4, 4))
            else:
                gt.append(query_camera.extrinsic.E)

            if estimated_camera is None:
                estm.append(np.zeros(4, 4))
            else:
                estm.append(estimated_camera.extrinsic.E)

            entries.append(entry)

        except Exception as e:
            print(traceback.format_exc())
    if opt.exp_name is None:
        exp_dir = f'exp/{datetime.datetime.now().isoformat()}'
    else:
        exp_dir = f'exp/{opt.exp_name}'
    os.makedirs(exp_dir, exist_ok=True)
    np.save(os.path.join(exp_dir, 'gt.npy'), np.asarray(gt))
    np.save(os.path.join(exp_dir, 'estm.npy'), np.asarray(estm))
    np.save(os.path.join(exp_dir, 'entries.npy'), np.asarray(entries))
