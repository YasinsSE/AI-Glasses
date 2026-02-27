# Dataset: SANPO

## About the Original SANPO Dataset
[SANPO](https://research.google/blog/sanpo-a-scene-understanding-accessibility-navigation-pathfinding-obstacle-avoidance-dataset/) (Scene understanding, Accessibility, Navigation, Pathfinding, Obstacle avoidance dataset) is a large-scale video dataset developed by Google Research. It is specifically designed for outdoor, egocentric (first-person) scene analysis and human navigation systems. The word "sanpo" translates to "brisk walk" or "stroll" in Japanese.

The original dataset was created to foster the development of autonomous robots, augmented reality (AR) smart glasses, and assistive technologies for the visually impaired. It contains over 112,000 real and 113,000 synthetic high-resolution frames, including stereo RGB images, depth maps, and panoptic/segmentation masks.

## ALAS Project Optimization
The original size of the SANPO dataset (~6TB) and its multi-sensor contents are too resource-heavy for direct use on embedded systems. Since the **ALAS Project** targets a real-time, fully offline, and monocular (single-camera) AI architecture running on an Nvidia Jetson Nano, we applied a strict custom filtering strategy to the dataset.

To conserve storage, reduce bandwidth, and optimize the training pipeline, the following data was explicitly **excluded** from our subset:
* `camera_chest` (Chest-mounted camera data)
* `right` (Right lens stereo images)
* `depth_maps` & `zed_depth_maps` (Pre-computed depth maps)
* `sanpo-synthetic` (All synthetic data)

Model training is conducted exclusively using the **`camera_head/left` (Left Head Camera)** RGB images and their corresponding ground-truth `segmentation_masks`.

---

## Directory Structure
The ALAS project utilizes this highly filtered version of the SANPO dataset. The customized directory structure is organized as follows:

> 🔗 **Class Mapping:** You can view the global 31-class dictionary directly here: [`labelmap.json`](./labelmap.json)

```text
sanpo_dataset/
├── labelmap.json (Global 31-class mapping dictionary)
├── all_candidates.txt (List of all session IDs that contain segmentation masks)
├── <SESSION_ID>/ (A unique folder representing a single recording session)
│   ├── description.json (Metadata: weather, traffic, and camera calibration)
│   └── camera_head/ (Data captured specifically from the head-mounted camera)
│       ├── camera_poses.csv (Dynamic spatial tracking data per frame)
│       ├── fixed_camera_poses.csv (Static or baseline calibration data)
│       └── left/ (Left lens data - exclusively used for model training)
│           ├── frame_segmentation_annotation_type.json (Indicates if annotation is human or machine generated)
│           ├── video_frames/ (Directory containing raw RGB input images)
│           │   ├── 000000.png
│           │   ├── 000001.png
│           │   └── ...
│           └── segmentation_masks/ (Directory containing ground-truth semantic masks)
│               ├── 000000.png
│               ├── 000001.png
│               └── ...
└── <OTHER_SESSION_IDs>/
    └── ...