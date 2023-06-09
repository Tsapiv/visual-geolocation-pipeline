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



## Dataset Collection and Processing
To facilitate the collection and processing of geotagged datasets for visual localization, we provide handy tools that leverage the Google Street View API. These tools simplify the gathering and preparation of data for use in the visual localization pipeline.

## Contributing
Contributions to enhance the functionality and performance of the visual localization pipeline are welcome. If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is licensed under the MIT License. See the LICENSE file for more information.

## Contact
For any questions or inquiries, please open issue.
