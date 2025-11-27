# ALAS – AI-Based Assistive Glasses for Visually Impaired Navigation

ALAS is a wearable system designed to assist visually impaired individuals during outdoor mobility by providing continuous environmental awareness through local, real-time processing.  
The system integrates RGB-D sensing, lightweight semantic segmentation, depth-based obstacle analysis, and offline navigation to deliver audio guidance without requiring internet connectivity.

## Problem Definition

Traditional mobility tools such as canes or phone-based navigation lack real-time obstacle detection and do not provide fine-grained environmental understanding.  
These limitations reduce safety and independence, especially in unfamiliar or dense urban environments.

ALAS aims to address this limitation by running perception, navigation, and audio interaction directly on embedded hardware, enabling safe mobility through fully local processing.

## System Concept

The system consists of three tightly coupled components:

### Hardware Layer
- RGB-D camera for spatial and visual perception  
- IMU for motion and orientation estimation  
- GPS for location awareness  
- Microphone and speaker for audio interaction  
- Button for controlled STT activation  
- Embedded compute unit (Raspberry Pi class device)  
- Battery and power-management module

### AI & Perception Layer
- Preprocessing of RGB-D frames  
- Lightweight U-Net-based or MobileNet-based segmentation models suitable for embedded inference  
- Pixel-level detection of walkable areas, curbs, and obstacles  
- Depth-assisted distance estimation and obstacle severity evaluation  
- Fusion of visual and sensor data into a unified perception output

### Interaction & Navigation Layer
- Offline Speech-to-Text for command recognition (button-activated)  
- Offline Text-to-Speech for continuous guidance  
- Route computation using preprocessed OpenStreetMap data  
- Dijkstra-based pathfinding adapted for pedestrian networks  
- Continuous localization using GPS and IMU fusion  
- Real-time adaptation of navigation and warning outputs

## Research Scope (Current Stage)

The initial phase focuses on understanding the technical landscape and constraints of embedded AI systems.  
Research includes:

- Analysis of segmentation architectures for low-power embedded devices  
- Study of RGB-D preprocessing pipelines and latency constraints  
- Evaluation of offline STT/TTS systems suitable for limited hardware  
- Examination of OSM-based routing and lightweight graph processing  
- Identification of sensor fusion requirements for robust outdoor navigation  
- Development of performance requirements and architectural boundaries

Findings from this phase will define the implementation strategy for the prototype.

## Implementation Outlook

After completing the research and literature review, the project will proceed to full prototype development, including:

- Training and optimization of a segmentation model for embedded inference  
- Integration of camera, IMU, GPS, audio IO, and processing pipeline  
- Embedded implementation of routing algorithms  
- Real-time fusion of perception and navigation outputs  
- System-level evaluation through controlled indoor and outdoor testing

The final objective is a portable, locally operating assistive device capable of perceiving the environment, interpreting scene structure, and guiding the user safely through audio feedback.