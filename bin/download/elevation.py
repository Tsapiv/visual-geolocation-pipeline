from argparse import ArgumentParser

from geonavpy.download.elevation import download_elevation

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to metadata file')
    parser.add_argument('--credentials', type=str, default='credentials/google.yaml', help='Path to metadata file')
    parser.add_argument('--chunk_size', type=int, default=300, help='Field of view')

    opt = parser.parse_args()

    download_elevation(opt.dataset, opt.credentials, opt.chunk_size)
