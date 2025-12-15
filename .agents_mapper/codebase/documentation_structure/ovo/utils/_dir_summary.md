# Directory Summary: OVO Utilities

This directory (`ovo/utils/`) provides a collection of general-purpose utility functions used across the OVO project. These functions handle common operations such as I/O, geometric calculations, instance manipulation, and evaluation metrics.

## Key Files:
*   `gen_utils.py`: General utility functions.
*   `io_utils.py`: Input/Output operations, such as loading and saving data.
*   `geometry_utils.py`: Functions for 3D geometric computations.
*   `instance_utils.py`: Critical functions for managing `Instance3D` objects, including the `same_instance` function for merging decisions.
*   `clip_utils.py`: Utility functions specifically for handling CLIP features.
*   `eval_utils.py`: Functions for evaluating the performance of the OVO system.
*   `segment_utils.py`: Utilities related to segmentation tasks.
*   `vis_utils.py`: Visualization utility functions.

## Purpose:
The `utils` directory consolidates common, reusable functionalities, promoting code reusability and keeping the core logic in `ovo/entities` cleaner. `instance_utils.py` is particularly important for the instance merging logic.