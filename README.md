# Project Overview: Spacecraft Module and Celestial Body Detection

This project focuses on designing and implementing a solution to enable a robotic system to detect windows in spacecraft modules, identify celestial bodies (specifically Earth and Moon), and navigate within a simulated environment. The system is divided into several key components, each addressing specific challenges encountered during the project.

## Table of Contents
1. [Problem Decomposition](#problem-decomposition)
2. [Key Components](#key-components)
   - [Window Detection](#window-detection)
   - [Planet Detection](#planet-detection)
     - [Stars Removal](#stars-removal)
     - [Planet Separation](#planet-separation)
     - [Template Matching](#template-matching)
     - [Pre-trained CNN Usage](#pre-trained-cnn-usage)
   - [Module Detection](#module-detection)
   - [Window Identification in Modules](#window-identification-in-modules)
     - [Window Search](#window-search)
     - [Moving to Windows](#moving-to-windows)
     - [Duplicate Window Handling](#duplicate-window-handling)
   - [Image Stitching](#image-stitching)
   - [Distance Calculations](#distance-calculations)
3. [Implementation and Results](#implementation-and-results)
   - [Simulation Performance](#simulation-performance)
   - [Limitations and Observations](#limitations-and-observations)
   - [Real Robot Testing](#real-robot-testing)
     - [Adaptation for Real-World Testing](#adaptation-for-real-world-testing)
     - [Real-World Performance](#real-world-performance)
4. [Conclusion](#conclusion)
5. [Video Demonstration](#video-demonstration)

## Problem Decomposition

To tackle the complex task of spacecraft module and celestial body detection, the problem was decomposed into manageable chunks, each focusing on different aspects of the system. This modular approach allowed for easier integration and testing of individual components.

## Key Components

### Window Detection

The system uses the robot's camera feed to detect windows within spacecraft modules. The detection algorithm relies on contour analysis, where contours below a certain area threshold are filtered out. Rectangularity is assumed, so the aspect ratio and bounding rectangle dimensions of the contours are analyzed to ensure they match typical window characteristics. This method, however, is sensitive to image quality and perspective distortions, which may lead to false positives or missed detections.

### Planet Detection

Once a window is detected, the system proceeds to identify celestial bodies inside the module.

#### Stars Removal

To enhance the accuracy of detecting planets using the Hough Transform, a preprocessing step removes stars from the image. A white mask identifies potential star regions, which are then dilated to cover and eliminate the stars, resulting in a cleaner image for subsequent circle detection.

#### Planet Separation

The system converts the preprocessed image to greyscale, applies Gaussian blur to reduce noise, and then uses the Hough Circle Transform to detect planets. Detected planets are stored as objects, and their images are cropped and saved for further analysis.

#### Template Matching

Initially, template matching was used to identify planets by comparing detected planets with predefined Earth and Moon templates. However, this method struggled with variations in image quality, lighting, and rotation, leading to the adoption of a more robust machine learning approach.

#### Pre-trained CNN Usage

To overcome the limitations of template matching, a pre-trained convolutional neural network (CNN), MobileNetV2, was fine-tuned with a custom dataset to identify planets. The model, trained with augmented data to improve generalization, demonstrated high accuracy in classifying celestial bodies under various conditions.

### Module Detection

The system uses the window detection component to locate and assess spacecraft modules. Initially, the robot navigates to the closest module entrance and performs a 360-degree scan to determine the module's safety based on the presence of green or red circles. This method, while adaptable, is time-consuming and may misclassify modules under certain conditions.

### Window Identification in Modules

#### Window Search

The robot employs a randomized goal selection strategy to search for windows within a module. It navigates to random points and performs 360-degree scans to detect windows. If a window is detected, the robot adjusts its position to capture a high-quality image before continuing the search.

#### Moving to Windows

Upon detecting a window, the robot calculates the necessary rotation to align with the window and then moves towards it. This process involves calculating pixel-to-degree ratios for accurate alignment and incorporating time-out functionality to prevent the robot from getting stuck.

#### Duplicate Window Handling

To avoid capturing duplicate windows, the system uses the Scale-Invariant Feature Transform (SIFT) algorithm to compare new window images with previously captured ones. This technique, although computationally expensive, effectively identifies and filters out duplicates based on visual similarity.

### Image Stitching

The detected images of Earth and Moon are stitched together using SIFT for keypoint detection and RANSAC for homography estimation. The resulting images are warped and blended to create a seamless panoramic view.

### Distance Calculations

The distance between Earth and Moon is calculated using triangulation, based on their detected positions in the images. While this method provides a simplified estimation, it assumes ideal conditions and may not be entirely accurate in real-world scenarios.

## Implementation and Results

### Simulation Performance

The system was tested in various simulated environments using Gazebo. It demonstrated high efficiency and accuracy in navigating and detecting celestial bodies across different scenarios, including an "Interesting" environment with additional obstacles. The robot consistently identified and captured images of the Earth and Moon, even under challenging conditions.

### Limitations and Observations

Despite the system's robustness, certain limitations were noted, such as occasional misclassification of objects due to image quality issues and the robot getting stuck during navigation. Additionally, the system's reliance on certain assumptions (e.g., consistent lighting, fixed perspective) may limit its effectiveness in more complex real-world scenarios.

### Real Robot Testing

#### Adaptation for Real-World Testing

Adapting the solution for real-world testing required addressing challenges like varying lighting conditions, physical obstacles, and differences in module size. Adjustments were made to the computer vision algorithms, data augmentation techniques were employed, and the system was fine-tuned to handle the unique constraints of the real environment.

#### Real-World Performance

In real-world testing, the robot successfully navigated to and identified the correct module, captured images of windows, and attempted to detect the Earth and Moon. However, challenges were encountered in accurately detecting celestial bodies, highlighting the need for further refinement.

## Video Demonstration

For a visual overview of the project, please refer to the [video demonstration](https://www.youtube.com/watch?v=UU7TQqW6gh0).

See the report for further detail. 
