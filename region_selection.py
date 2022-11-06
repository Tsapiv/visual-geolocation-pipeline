import json

import matplotlib

matplotlib.use('TkAgg')
import osmnx as ox
from argparse import ArgumentParser

POINT = (49.8440167, 24.0240236)  # (49.8348382, 24.0348626)


def create_subgraph_v1(g, point, max_dist=1000):
    center = ox.nearest_nodes(g, *point[::-1])
    roi = ox.truncate_graph_dist(g, center, max_dist=max_dist)
    return roi, center


def create_subgraph_v2(g, point, max_dist=1000):
    polygon = ox.bbox_to_poly(*ox.bbox_from_point(point, dist=max_dist))
    roi = ox.truncate_graph_polygon(g, polygon)
    return roi, ox.nearest_nodes(g, *point[::-1])


def get_coords_around_point(point, radius, visualize=False):
    g_city = ox.graph_from_place('Lviv', network_type='drive', simplify=True)

    roi, center = create_subgraph_v2(g_city, point, radius)
    coords = []
    for node_id in roi.nodes:
        node = roi.nodes[node_id]
        coords.append([node['y'], node['x']])

    if visualize:
        roi_nodes = set(roi.nodes)
        nc = []
        ns = []
        for node in g_city.nodes:
            if node == center:
                nc.append('g')
                ns.append(100)
            elif node in roi_nodes:
                nc.append('r')
                ns.append(15)
            else:
                nc.append('b')
                ns.append(5)

        ox.plot_graph(g_city, node_color=nc, node_size=ns, figsize=(12, 12), bgcolor="w")

    return coords


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', nargs='+', type=float, required=True, help='Define point')
    parser.add_argument('-r', type=float, default=1000, help='Radius in meters')
    parser.add_argument('-v', action='store_true', help='Option for visualization')
    parser.add_argument('-s', default=None, type=str, help='Option for save')

    args = parser.parse_args()

    coords = get_coords_around_point(tuple(args.p), args.r, args.v)
    if args.s:
        json.dump(coords, open(args.s, 'w'), indent=4)
