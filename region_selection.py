import json

import matplotlib
import numpy as np
import pyproj
from matplotlib import pyplot as plt
from shapely import ops
from shapely.geometry import LineString
from shapely.geometry.multilinestring import MultiLineString
from shapely.ops import transform

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

def redistribute_vertices(geom, distance):
    if geom.geom_type == 'LineString':
        num_vert = max(int(round(geom.length / distance)), 1)
        coords = [geom.interpolate(float(n) / num_vert, normalized=True)
             for n in range(0, num_vert + 1)]
        if len(coords) < 2:
            return None
        return LineString(coords)
    elif geom.geom_type == 'MultiLineString':
        parts = [redistribute_vertices(part, distance)
                 for part in geom.geoms]
        return type(geom)([p for p in parts if p is not None and not p.is_empty])
    else:
        raise ValueError('unhandled geometry %s', (geom.geom_type,))


def get_coords_around_point(point, radius, spacing, visualize=False):
    g_city = ox.graph_from_place('Lviv', network_type='drive', simplify=False)
    subgraph, center = create_subgraph_v2(g_city, point, radius)
    coords = []
    geodesic = pyproj.Geod(ellps='WGS84')





    # for u_id, v_id, _ in subgraph.edges:
    #     u = subgraph.nodes[u_id]
    #     v = subgraph.nodes[v_id]
    #     d = geodesic.inv(u['y'], u['x'], v['y'], v['x'])[-1]
    #     if d <= spacing:
    #         # coords.append([u['y'], u['x']])
    #         continue
    #
    #     r = geodesic.inv_intermediate(u['y'], u['x'], v['y'], v['x'], del_s=spacing, initial_idx=0, terminus_idx=0)
    #
    #     coords.extend(list(zip(r.lons, r.lats)))


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

    resampled_multi_line_utm = redistribute_vertices(multi_line_utm, 30)

    project_utm_to_wgs84 = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True).transform
    resampled_multi_line_wgs84: MultiLineString = transform(project_utm_to_wgs84, resampled_multi_line_utm)

    for linestring in resampled_multi_line_wgs84.geoms:
        coords.extend(np.asarray(linestring.coords.xy).T.tolist())

    if visualize:
        fig, axis = ox.plot_graph(g_city, node_size=1, show=False, figsize=(12, 12), bgcolor="w")
        plt.scatter(*(list(zip(*coords))[::-1]), c='b', marker='o', s=1)
        plt.show()
    return coords


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', nargs='+', type=float, required=True, help='Define point')
    parser.add_argument('-r', type=float, default=1000, help='Radius in meters')
    parser.add_argument('-v', action='store_true', help='Option for visualization')
    parser.add_argument('-s', default=None, type=str, help='Option for save')
    parser.add_argument('-d', default=30, type=int, help='Distance between coordinates in meters')


    opt = parser.parse_args()

    coords = get_coords_around_point(tuple(opt.p), opt.r, opt.d, opt.v)
    print(f'Total coordinates number: {len(coords)}')
    if opt.s:
        json.dump(coords, open(opt.s, 'w'), indent=4)
