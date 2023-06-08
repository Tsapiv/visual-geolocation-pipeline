from typing import Union, Tuple, Optional

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


def _create_subgraph_v1(g, point, max_dist=1000):
    center = ox.nearest_nodes(g, *point[::-1])
    roi = ox.truncate_graph_dist(g, center, max_dist=max_dist)
    return roi, center


def _create_subgraph_v2(g, point, max_dist=1000):
    polygon = ox.bbox_to_poly(*ox.bbox_from_point(point, dist=max_dist))
    roi = ox.truncate_graph_polygon(g, polygon)
    return roi, ox.nearest_nodes(g, *point[::-1])


def _redistribute_vertices(geom, distance, threshold_coeff=1.5):
    coords = []
    for part in geom.geoms:
        assert part.geom_type == 'LineString'
        n_vert = int(part.length / distance)
        if part.length < distance / threshold_coeff:
            continue
        coords.append(Point(part.coords[0]))
        coords.append(Point(part.coords[-1]))
        for i in range(1, n_vert):
            coords.append(part.interpolate(i / n_vert, normalized=True))
    multipoint = MultiPoint(coords)
    return multipoint


def _get_graph(query: Union[str, Tuple[int, int]], radius: Optional[int]):
    if isinstance(query, str):
        return ox.graph_from_place(query, network_type='walk', simplify=False)
    elif isinstance(query, (list, tuple)) and radius is not None:
        return ox.graph_from_point(query, dist=radius, network_type='walk', simplify=False)
    raise ValueError('Invalid parameters to retrieve city network')


def sample_lats_lngs(query: Union[str, Tuple[int, int]], spacing: float, radius: Optional[int] = None,
                     jitter: Optional[Tuple[float]] = None, visualize: bool = False):
    graph = _get_graph(query, radius)

    coords = []
    lines = []

    for u_id, v_id, _ in graph.edges:
        u = graph.nodes[u_id]
        v = graph.nodes[v_id]
        line = LineString([[u['y'], u['x']], [v['y'], v['x']]])
        lines.append(line)

    multi_line = MultiLineString(lines)

    multi_line_wgs84: MultiLineString = ops.linemerge(multi_line)

    wgs84 = pyproj.CRS('EPSG:4326')
    utm = pyproj.CRS('EPSG:32635')

    project_wgs84_to_utm = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
    multi_line_utm = transform(project_wgs84_to_utm, multi_line_wgs84)

    resampled_multi_line_utm = _redistribute_vertices(multi_line_utm, spacing)

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
        ox.plot_graph(graph, node_size=0, show=False, figsize=(12, 12), bgcolor="w")
        plt.scatter(*(list(zip(*coords))[::-1]), c='b', marker='o', s=1)
        plt.show()
    return coords
