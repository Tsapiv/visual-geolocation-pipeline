import os
import re
from argparse import ArgumentParser
from typing import List

import matplotlib
import polyline
import pyproj
import requests
import yaml
from tqdm import tqdm

from download_region_data import META_REQUEST, DATA_REQUEST
from utils import sign_url

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
import json

if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument('--input', required=True, type=str, help='Prefix to *_features.npy and *_id.json')
    parser.add_argument('-k', required=False, type=int, default=3)
    parser.add_argument('-v', action='store_true', help='Option for visualization')

    opt = parser.parse_args()

    threshold = 5

    route_pack = json.load(
        open('data/route-49.85170508786705,24.0199599920372-49.83973996916215,24.01095725864545.json'))

    g_city = ox.graph_from_place('Lviv', network_type='drive')
    print('Finish loading')

    route = polyline.decode(route_pack['routes'][0]['overview_polyline']['points'])

    nodes, displacement = ox.nearest_nodes(g_city, *(list(zip(*route))[::-1]), return_dist=True)

    # print(len(nodes[0]))

    nodes = list(map(lambda y: y[0], filter(lambda x: x[-1] < threshold, zip(nodes, displacement))))

    # print(len(n))

    paths = [ox.shortest_path(g_city, nodes[i - 1], nodes[i]) for i in range(1, len(nodes))]

    geodesic = pyproj.Geod(ellps='WGS84')
    for i in range(1, len(route)):
        print(route[i])
        heading, _, _ = geodesic.inv(*route[i - 1], *route[i])

        heading %= 360



        print(heading)
        # print(g_city.nodes[nodes[i - 1]])

    # ox.plot_graph_routes(g_city, paths, route_colors='b', route_linewidths=2)
    p = 'route-49.85170508786705,24.0199599920372-49.83973996916215,24.01095725864545'
    cred = yaml.safe_load(open('credentials/google.yaml'))
    root = os.path.join('data', p)
    os.makedirs(os.path.join(root, 'photo'))
    meta = []
    for idx in tqdm(range(0, len(route) - 1)):
        try:
            heading = geodesic.inv(*route[idx], *route[idx + 1])[0] % 360
            params = dict(location=",".join(map(str, route[idx])), size='640x640', heading=0, key=cred['api-key'])
            params['heading'] = heading
            request = DATA_REQUEST.format(**params)
            request = sign_url(request, cred['secret'])
            response = requests.get(request)
            if not response.ok:
                print(response.status_code)
                continue
            with open(os.path.join(root, 'photo', f'{idx}_{params["location"]}_{heading}.jpg'), 'wb') as file:
                file.write(response.content)
            response.close()
        except Exception as e:
            print(e)
