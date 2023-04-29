import json
import os
from argparse import ArgumentParser

from camera import CameraMetadata
from dataset import Dataset
from geoutils import intrinsics_from_metadata, relative_extrinsic_from_metadata

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', required=True, type=str, help='Path to dataset')

    opt = parser.parse_args()

    db = Dataset(opt.input)

    metadata = [CameraMetadata.from_kwargs(**m) for m in db.metadata(db.entries, cache=True)]

    anchor = CameraMetadata.from_kwargs(**{"alt": 295.0958251953125,
                                           "lat": 49.84670475881484,
                                           "lng": 24.03118185004758,
                                           "azn": 0.0,
                                           "h": 640,
                                           "w": 640})

    for uuid, m in zip(db.entries, metadata):
        K = intrinsics_from_metadata(m).K
        E = relative_extrinsic_from_metadata(anchor, m).E

        base = db.metadata(uuid)
        base.update(K=K.tolist(), E=E.tolist(), basealt=anchor.alt, baselat=anchor.lat, baselng=anchor.lng,
                    baseazn=anchor.azn)

        with open(os.path.join(db.root, uuid, 'metadata.json'), 'w') as f:
            json.dump(base, f, indent=4)
