# A deep learning-based pipeline for visual geolocation in the urban environment

This repository contains the implementation of a visual localization pipeline for city-scale environments. The project explores various approaches to visual localization and evaluates their pros and cons to build the most suitable city-scale visual localization pipeline. The main objective of this solution is to robustly and accurately estimate the camera position based on a single image and camera parameters that are usually automatically attached to the image through EXIF.

### Features
- Robust and accurate camera position estimation based on a single image and camera parameters.
- Flexibility and extensibility to accommodate the growth of the reference image database without the need for retraining.
- Handy tools for collecting and processing geotagged datasets for visual localization based on `Google Street View API`.
## Repository Contents

- `src/geonavpy` Python Package: Contains the algorithms and modules for the visual localization pipeline.
- `bin/` Directory: Includes scripts for generating the reference database and running experiments.

## Installation

1. Clone the repository:
```shell
git clone https://github.com/Tsapiv/visual-localization-pipeline.git
cd visual-localization-pipeline
```
2. Install the required dependencies:
```shell
pip install poetry
poetry install
```

## Usage
3. Run experiments:
```shell
python bin/run_experiments.py --reference_set <reference_set_path> --descriptor_type <descriptor_type> --query_set <query_set_path> --exp_name <experiment_name> --conf <config_file_path>
```
- `--reference_set`: Path to the reference dataset.
- `--descriptor_type`: Type of global descriptor used in retrieval (default: `radenovic_gldv1`).
- `--query_set`: Path to the query dataset.
- `--exp_name`: Experiment name (optional).
- `--conf`: Path to the config file.

4. Generate reference database:
    
    4.1. Preview location and adjust spacing using OpenStreetMap API
    ```shell
    python bin/preview.py [-h] (--point POINT POINT | --place PLACE) [--radius RADIUS] [--spacing SPACING] [--jitter LOWER_JITTER UPPER_JITTER] [-v]
    ```
    To generate coordinates around a specific point:
    ```shell
    python bin/preview.py --point 40.7128 -74.0060 --radius 1000 --spacing 50 -v
    ```
    To generate coordinates around a specific city:
    ```shell
    python bin/preview.py --place New York --radius 2000 --jitter 1 2
    ```
   
    4.2. Download metadata of actual panoramas using Google Street View API (free)
    ```shell
    python bin/download/metadata.py [-h] --database DATABASE [--credentials CREDENTIALS] (--point POINT POINT | --place PLACE) [--jitter LOWER_JITTER UPPER_JITTER] [--radius RADIUS] [--spacing SPACING] [-v]
    ```
    Use the same parameters as in the "bin/preview.py" script
   
    4.3. Generate html map with positions of actual panoramas
    ```shell
    python bin/postview.py [-h] --database DATABASE [--credentials CREDENTIALS] [--zoom ZOOM] [--output OUTPUT]
    ```
    Use the same "database" parameter as in the "bin/download/metadata.py" script

    4.4. Update metadata files with altitude info using Google Elevation API (priced)
    ```shell
    python bin/download/elevation.py [-h] --database DATABASE [--credentials CREDENTIALS]
    ```

    4.5. Download panoramas and update metadata files using Google Street View API (priced)
    ```shell
    python bin/download/panoramas.py [-h] --database DATABASE [--credentials CREDENTIALS] [--fov FOV] [--n_directions N_DIRECTIONS]
    ```
   
    4.6. Precompute global feature descriptors based on panoramas for image retrieval
    ```shell
    python bin/generate_global_descriptors.py [-h] --database DATABASE [--descriptor_type DESCRIPTOR_TYPE] [--backbone BACKBONE] [--device DEVICE]
    ```

    4.7. Precompute local feature descriptors based on panoramas for feature matching
    ```shell
    python bin/generate_local_descriptors.py [-h] --database DATABASE [--max_keypoints MAX_KEYPOINTS] [--keypoint_threshold KEYPOINT_THRESHOLD] [--nms_radius NMS_RADIUS] [--device DEVICE]
    ```

## Database Structure
```
.
└── <database name>
    ├── <database entry uid 1>
    │   ├── image.jpg
    │   ├── metadata.json
    │   │   ├── "w": int,
    │   │   ├── "h": int,
    │   │   ├── "lat": Optional[float], # latitude
    │   │   ├── "lng": Optional[float], # longitude
    │   │   ├── "alt": Optional[float], # altitude
    │   │   ├── "azn": Optional[float], # azimuth
    │   │   ├── "fov": Optional[float], # camera's field of view 
    │   │   ├── "K": 3x3 float matrix # intrinsic camera calibration
    │   │   └── "E": Optional[4x4 float matrix] # extrinsic camera calibration
    │   ├── [<optional model specifier prefix>]keypoints.npy
    │   ├── <model specifier prefix 1>_descriptor.npy
    │   ├── <model specifier prefix 2>_descriptor.npy
    │   ├── ....
    │   └── <model specifier prefix k>_descriptor.npz
    ├── <database entry uid 2>
    ├── <database entry uid 3>
    ├── <database entry uid 4>
    ├── ....
    └── <database entry uid n>
```
## Contributing
Contributions to enhance the functionality and performance of the visual localization pipeline are welcome. If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## Credits
Image retrieval: [Deep Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark)

Image reranking: [Understanding Image Retrieval Re-Ranking: A Graph Neural Network Perspective](https://github.com/Xuanmeng-Zhang/gnn-re-ranking)

Feature matching: [SuperPoint & SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)

## License
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is licensed under the MIT License. See the LICENSE file for more information.

## Contact
For any questions or inquiries, please open issue.
