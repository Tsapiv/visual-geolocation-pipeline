import json
import os
from collections import defaultdict

from dataset import Dataset

if __name__ == '__main__':

    dataset = Dataset('datasets/Lviv49.8443931@24.0254815')

    meta = json.load(open('data/49.8443931@24.0254815/metadata.json'))

    print(len(meta), len(dataset.entries) / 8)

    c1 = defaultdict(list)
    for entry in dataset.entries:
        m = dataset.metadata(entry)
        c1[(m['lat'], m['lng'])].append(entry)

    c2 = {}
    for m in meta:
        try:
            c2[(m['location']['lat'], m['location']['lng'])] = m['pano_id']
        except:
            pass

    print(len(c1), len(c2))

    c3 = {c2[k]: v for k, v in c1.items()}

    for p, es in c3.items():
        for i, m in enumerate(dataset.metadata(es)):

            m.update(pano=p)
            json.dump(m, open(os.path.join(dataset.root, es[i], 'metadata.json'), 'w'), indent=4)