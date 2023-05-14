import json
import time
from argparse import ArgumentParser

import matplotlib
import numpy as np
import osmnx as ox
import pyproj
from matplotlib import pyplot as plt
from pyproj import Geod
from shapely import ops
from shapely.geometry import LineString, MultiPoint, Point
from shapely.geometry.multilinestring import MultiLineString
from shapely.ops import transform

matplotlib.use('TkAgg')


def create_subgraph_v1(g, point, max_dist=1000):
    center = ox.nearest_nodes(g, *point[::-1])
    roi = ox.truncate_graph_dist(g, center, max_dist=max_dist)
    return roi, center


def create_subgraph_v2(g, point, max_dist=1000):
    polygon = ox.bbox_to_poly(*ox.bbox_from_point(point, dist=max_dist))
    roi = ox.truncate_graph_polygon(g, polygon)
    return roi, ox.nearest_nodes(g, *point[::-1])


def redistribute_vertices(geom, distance):
    coords = []
    for part in geom.geoms:
        assert part.geom_type == 'LineString'
        n_vert = int(part.length / distance)
        if part.length < distance / 1.5:
            continue
        coords.append(Point(part.coords[0]))
        coords.append(Point(part.coords[-1]))
        for i in range(1, n_vert):
            coords.append(part.interpolate(i / n_vert, normalized=True))
    multipoint = MultiPoint(coords)
    return multipoint


def get_coords_around_point(point, radius, spacing, jitter=None, visualize=False):
    g_city = ox.graph_from_place('Lviv', network_type='walk', simplify=False)
    # ox.save_graphml(g_city, 'data/graph3.graphml')
    t = time.time()
    # g_city = ox.load_graphml('data/graph3.graphml')
    print(time.time() - t)
    # fig, axis = ox.plot_graph(g_city, node_size=1, show=True, figsize=(12, 12), bgcolor="w")

    subgraph, center = create_subgraph_v2(g_city, point, radius)
    coords = []
    lines = []

    for u_id, v_id, _ in subgraph.edges:
        u = subgraph.nodes[u_id]
        v = subgraph.nodes[v_id]
        line = LineString([[u['y'], u['x']], [v['y'], v['x']]])
        lines.append(line)

    multi_line = MultiLineString(lines)

    multi_line_wgs84: MultiLineString = ops.linemerge(multi_line)

    wgs84 = pyproj.CRS('EPSG:4326')
    utm = pyproj.CRS('EPSG:32635')

    project_wgs84_to_utm = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
    multi_line_utm = transform(project_wgs84_to_utm, multi_line_wgs84)

    resampled_multi_line_utm = redistribute_vertices(multi_line_utm, spacing)

    project_utm_to_wgs84 = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True).transform
    resampled_multi_line_wgs84: MultiLineString = transform(project_utm_to_wgs84, resampled_multi_line_utm)

    for linestring in resampled_multi_line_wgs84.geoms:
        coords.extend(np.asarray(linestring.coords.xy).T.tolist())

    coords = np.asarray(list(set(map(tuple, coords))))

    if jitter is not None:
        dist = np.random.uniform(*jitter, len(coords))
        az = np.random.uniform(0, 360, len(coords))
        g = Geod(ellps='WGS84')
        lng, lat, _ = g.fwd(coords[:, 1], coords[:, 0], az, dist)

        coords = np.concatenate([lat.reshape(-1, 1), lng.reshape(-1, 1)], axis=-1)

    if visualize:
        fig, axis = ox.plot_graph(g_city, node_size=0, show=False, figsize=(12, 12), bgcolor="w")
        plt.scatter(*(list(zip(*coords))[::-1]), c='b', marker='o', s=1)
        plt.show()
    return coords


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', nargs='+', type=float, required=True, help='Define point')
    parser.add_argument('-j', nargs='+', type=float, default=None, help='Jitter range in meters')
    parser.add_argument('-r', type=float, default=1000, help='Radius in meters')
    parser.add_argument('-v', action='store_true', help='Option for visualization')
    parser.add_argument('-s', default=None, type=str, help='Option for save')
    parser.add_argument('-d', default=30, type=int, help='Distance between coordinates in meters')

    opt = parser.parse_args()

    jitter = tuple(opt.j) if opt.j is not None else None
    coords = get_coords_around_point(tuple(opt.p), opt.r, opt.d, jitter=jitter, visualize=opt.v)
    print(f'Total coordinates number: {len(coords)}')
    if opt.s:
        json.dump(coords, open(opt.s, 'w'), indent=4)
