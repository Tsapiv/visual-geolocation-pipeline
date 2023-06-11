from argparse import ArgumentParser

from geonavpy.download.elevation import download_elevation

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--database', type=str, required=True, help='Path to metadata file')
    parser.add_argument('--credentials', type=str, default='credentials/google.yaml', help='Path to metadata file')
    parser.add_argument('--chunk_size', type=int, default=300, help='Size of single request')

    opt = parser.parse_args()

    download_elevation(opt.database, opt.credentials, opt.chunk_size)
