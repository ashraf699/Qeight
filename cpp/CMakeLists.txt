cmake_minimum_required(VERSION 3.21)

project(qeight_pipeline)

# ---------------------------------------------------------
# C++ Standard
# ---------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------
# Compiler optimizations
# ---------------------------------------------------------

add_compile_options(
        -O3
        -march=armv8-a+simd
        -mtune=cortex-a55
        -ffast-math
        -funroll-loops
        -fPIC
        -fvisibility=hidden
        -fvisibility-inlines-hidden
        -ffunction-sections
        -fdata-sections
)

# ---------------------------------------------------------
# Linker dead-code stripping
# ---------------------------------------------------------

add_link_options(
        -Wl,--gc-sections
        -Wl,--strip-all
)

# ---------------------------------------------------------
# OpenCV include path
# ---------------------------------------------------------

include_directories(
        ${CMAKE_SOURCE_DIR}/opencv/include
)

# ---------------------------------------------------------
# OpenCV static libraries directory
# ---------------------------------------------------------

set(OPENCV_LIB_DIR
        ${CMAKE_SOURCE_DIR}/opencv/libs/arm64-v8a
)

# ---------------------------------------------------------
# Native shared library
# ---------------------------------------------------------

add_library(qeight_pipeline SHARED
        QeightJNI.cpp
        PipelineEngine.cpp
        OverlayRenderer.cpp
        PoolPhysicsEngine.cpp
        TrajectoryPhysicsEngine.cpp
        IndirectShotSolver.cpp
        IndirectPhysics.cpp
)

# ---------------------------------------------------------
# Android system libraries
# ---------------------------------------------------------

find_library(log-lib log)
find_library(android-lib android)
find_library(jnigraphics-lib jnigraphics)
find_library(vulkan-lib vulkan)
find_library(egl-lib EGL)
find_library(glesv3-lib GLESv3)
find_library(m-lib m)
find_library(z-lib z)

# ---------------------------------------------------------
# Link libraries
# ---------------------------------------------------------

target_link_libraries(qeight_pipeline

        # OpenCV
        ${OPENCV_LIB_DIR}/libopencv_imgproc.a
        ${OPENCV_LIB_DIR}/libopencv_core.a

        # OpenCV dependencies
        ${OPENCV_LIB_DIR}/libtegra_hal.a
        ${OPENCV_LIB_DIR}/libcpufeatures.a

        # Android libraries
        ${android-lib}
        ${jnigraphics-lib}
        ${log-lib}
        ${vulkan-lib}
        ${egl-lib}
        ${glesv3-lib}
        ${m-lib}
        ${z-lib}
        atomic
)