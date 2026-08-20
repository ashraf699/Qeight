# Qeight — OpenCV powered real time 8 Ball Pool Aim Assist for Android

**Qeight** is a production-ready Android overlay application that provides real-time aim assistance for 8 Ball Pool (Miniclip). It uses advanced computer vision algorithms (OpenCV), GPU-accelerated compute shaders (Vulkan), and precise geometric calculations to detect balls, analyze shot angles, and render trajectory overlays.

---

## Features

- **Real-time ball detection** via HSV color thresholding and Hough circle detection
- **Cue alignment (CA) strip detection** using PCA on white pixel clusters
- **GPU-accelerated skeleton thinning** via GuoHall algorithm (Vulkan compute)
- **Fast line detection (FLD)** for detecting aim lines on the felt surface
- **AT strip generation** with circle-axis alignment and programmatic fallback
- **Reflection geometry** for cushion shots (up to 8 reflections)
- **Pocket detection and avoidance** with configurable pocket radius and N/S shift
- **Transparent overlay rendering** using Vulkan graphics pipelines
- **Calibration UI** for precise ROI adjustment with real-time preview
- **Floating settings panel** with live parameter adjustment (reflections, color, thickness)
- **Custom splash screen** with animated letter spacing and glowing '8' background
- **Full permission management** (overlay, MediaProjection, foreground service)

---

## Build Requirements

### Software

- **Android Studio** Hedgehog (2023.1.1) or later
- **Android SDK** API 35 (Android 15)
- **Android NDK** r25c or later
- **CMake** 3.21.0 or later
- **Kotlin** 1.9.22
- **Gradle** 8.2

### Hardware

- **Target device:** Realme 11 5G (RMX3780) or equivalent ARM64 device
- **Minimum API:** 29 (Android 10)
- **Target API:** 35 (Android 15)
- **ABI:** `arm64-v8a` exclusively

---

## OpenCV Setup

Qeight uses **OpenCV 4.9.0** with **opencv_contrib 4.9.0** compiled from source as **static libraries** (no dynamic `.so` files in APK).

### Build OpenCV for Android

1. Clone OpenCV and opencv_contrib:
   ```bash
   git clone --branch 4.9.0 https://github.com/opencv/opencv.git
   git clone --branch 4.9.0 https://github.com/opencv/opencv_contrib.git

### Project Structure

Qeight/
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── cpp/
│   │   │   │   ├── CMakeLists.txt
│   │   │   │   ├── QeightJNI.cpp
│   │   │   │   ├── PipelineEngine.cpp
│   │   │   │   ├── PipelineEngine.h
│   │   │   │   ├── VulkanCompute.cpp
│   │   │   │   ├── VulkanCompute.h
│   │   │   │   ├── OverlayRenderer.cpp
│   │   │   │   ├── OverlayRenderer.h
│   │   │   │   └── opencv/
│   │   │   │       ├── include/   (OpenCV headers)
│   │   │   │       └── libs/arm64-v8a/  (OpenCV static .a files)
│   │   │   ├── java/com/ashraf/qeight/
│   │   │   │   ├── MainActivity.kt
│   │   │   │   ├── OverlayService.kt
│   │   │   │   ├── ScreenCaptureManager.kt
│   │   │   │   └── CalibrationManager.kt
│   │   │   ├── res/
│   │   │   │   ├── layout/activity_main.xml
│   │   │   │   ├── values/colors.xml
│   │   │   │   ├── values/strings.xml
│   │   │   │   ├── values/themes.xml
│   │   │   │   ├── drawable/
│   │   │   │   └── mipmap-*/
│   │   │   └── AndroidManifest.xml
│   │   └── assets/splash_logo.png (optional)
│   └── build.gradle.kts
├── build.gradle.kts
├── settings.gradle.kts
├── gradle.properties
└── README.md
### Building
Open the project in Android Studio
Sync Gradle files
Build → Make Project
Run on device (physical device required — MediaProjection does not work in emulators)
Usage
Launch Qeight
Grant Overlay Permission and Screen Capture permissions
Tap CALIBRATE to adjust ROI (region of interest) for your device resolution
Tap START — Qeight will launch 8 Ball Pool and activate the overlay
Play normally — aim-assist rays appear automatically when you pull back the cue
Use the floating ≡ menu button to adjust:
CBC Reflections (0–8)
TGT Reflections (0–8)
Line Thickness (1–8px)
Cushion Shots (toggle)
Overlay Color (HSV picker)
Tap STOP to exit overlay mode
### License
Proprietary and Confidential

This software is the exclusive property of Ashraf. Unauthorized copying, distribution, modification, or use of this software, in whole or in part, is strictly prohibited without express written permission from the author.

© 2026 Ashraf. All rights reserved.

### Credits
Created by the Mastermind : Ashraf

Pipeline algorithm design, C++ reference implementation, Android port, Vulkan compute/graphics integration, UI/UX design, calibration system, and all documentation.

### Support
For issues, feature requests, or inquiries, contact: martinfranzese15@gmail.com
