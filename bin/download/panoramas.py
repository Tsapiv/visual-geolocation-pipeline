from argparse import ArgumentParser

from geonavpy.download.panoramas import download_panoramas

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to metadata file')
    parser.add_argument('--credentials', type=str, default='credentials/google.yaml', help='Path to metadata file')
    parser.add_argument('--fov', type=int, default=90, help='Field of view')
    parser.add_argument('--n_directions', type=int, default=8, help='Number of headings')

    opt = parser.parse_args()

    download_panoramas(opt.dataset, opt.credentials, opt.fov, opt.n_directions)
