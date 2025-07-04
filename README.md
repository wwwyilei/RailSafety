# Railway_Safety_system


## Graduate Project Overview
This project implements an advanced railway safety monitoring system using computer vision and deep learning techniques. The system is designed to detect and analyze near-miss incidents in railway environments using monocular camera footage, with a particular focus on accurate lateral distance estimation between objects and railway tracks.
Key Features

Real-time object detection and tracking using YOLO-World
Precise instance segmentation with EdgeSAM
Track detection and analysis using TEP-Net
Depth estimation using Depth Anything V2
Novel two-stage depth calibration method
Accurate lateral distance measurement

## System Requirements

Python 3.8+
PyTorch 1.10+
CUDA 11.3+ (for GPU acceleration)
16GB+ RAM recommended
NVIDIA GPU with 8GB+ VRAM recommended

## Dataset
The project uses a custom railway dataset with precise lateral distance annotations. You can access the dataset here:

Dataset link: 
https://tuenl-my.sharepoint.com/:f:/g/personal/w_y_l_wang_student_tue_nl/EuPBA71o3UhOn8bQTCEtVO0BqZY1kuIGNqSvKsGcrvV1qg?e=RUmxyQ


## Pre-trained Models
Pre-trained weights for the models can be downloaded from:  
Model Weights : 

https://tuenl-my.sharepoint.com/:f:/g/personal/w_y_l_wang_student_tue_nl/EsjFizFPFl1Lt_sUo15dYAcBMoPz-behe3lkPJITqr7U4w?e=K05cW7
s


Download and place the weights in the weights/ directory.

## Download Depthanything V2

https://tuenl-my.sharepoint.com/:f:/g/personal/w_y_l_wang_student_tue_nl/EsNkRocMXWdHpkbbnrsefUYBRVYAYg23csUFC0OKwBI40Q?e=aY2zCm

