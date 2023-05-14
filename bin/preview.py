from argparse import ArgumentParser
from pysight.mapping import sample_lats_lngs

if __name__ == '__main__':
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--point', nargs=2, type=float, help='Define point')
    group.add_argument('--place', type=str, help='City name')
    parser.add_argument('--jitter', nargs=2, type=float, default=None, help='Jitter range in meters')
    parser.add_argument('--radius', type=float, default=None, help='Radius in meters')
    parser.add_argument('--spacing', default=30, type=int, help='Distance between coordinates in meters')
    parser.add_argument('-v', action='store_true', help='Option for visualization')

    opt = parser.parse_args()
    query = tuple(opt.point) if opt.point is not None else opt.place
    jitter = tuple(opt.jitter) if opt.jitter is not None else None
    coords = sample_lats_lngs(query, opt.spacing, opt.radius, jitter, visualize=opt.v)
    print(f'Total coordinates number: {len(coords)}')
