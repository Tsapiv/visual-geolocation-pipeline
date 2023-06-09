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
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
    parser.add_argument('--descriptor_type', type=str, default='radenovic_gldv1', help='Path to reference dataset')
    parser.add_argument('--reference_entries', type=str, default=None, help='Path to query dataset')
    parser.add_argument('--query_entries', type=str, required=True, help='Path to query dataset')
    parser.add_argument('--conf', type=str, required=True, help='Path to config file')

    opt = parser.parse_args()

    conf = yaml.safe_load(open(opt.conf))

    dataset = Dataset(root=opt.dataset, descriptor_type=opt.descriptor_type)


    query_entries = np.load(opt.query_entries)

    if opt.reference_entries is not None:
        reference_entries = np.load(opt.reference_entries)
    else:
        reference_entries = set(dataset.entries) - set(query_entries)

    reference_set = dataset.get_subset(reference_entries)
    query_set = dataset.get_subset(query_entries)

    localizer = Localizer(conf, reference_set)

    for entry in tqdm(query_set.entries):
        try:
            query_image = query_set.image(entry)
            query_camera = camera_from_metadata(query_set.metadata(entry))

            # skip step with descriptor calculation
            query_descriptor = query_set.descriptor(entry)

            estimated_camera = localizer.localize(query_image, query_camera)


        except Exception as e:
            print(traceback.format_exc())
