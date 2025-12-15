# Directory Summary: Third-Party Integrations

This directory (`thirdParty/`) houses external libraries and modules that the OVO project integrates. These are typically not developed as part of OVO but are essential dependencies or provide specific functionalities that are leveraged by the main system.

## Key Subdirectories:
*   `ORB_SLAM3/`: Contains the ORB-SLAM3 library, used as a SLAM backbone for camera pose estimation and map building.
*   `segment-anything-2/`: Contains the Segment Anything 2 (SAM2) model, used for generating 2D instance masks from images.

## Purpose:
The `thirdParty` directory centralizes external dependencies, making it clear which parts of the codebase originate from outside the OVO project and how they are structured. It allows OVO to benefit from state-of-the-art research in SLAM and segmentation.