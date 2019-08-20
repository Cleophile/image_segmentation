cmake_minimum_required(VERSION 3.5)
SET(PROJ_NAME spix)
PROJECT(${PROJ_NAME})

ADD_DEFINITIONS(-std=c++11)

INCLUDE_DIRECTORIES(
    /usr/local/include
    /opt/ros/kinetic/include/
    /opt/ros/kinetic/include/opencv-3.3.1-dev
    )

LINK_DIRECTORIES(
    /opt/ros/kinetic/lib/x86_64-linux-gnu/
    )

# FIND_PACKAGE(OpenCV 3.3.0 REQUIRED )

ADD_EXECUTABLE(
    ${PROJ_NAME}
    main.cpp
    MBS.cpp
    )

TARGET_LINK_LIBRARIES(${PROJ_NAME}
    optimized libopencv_imgproc3.so
    optimized libopencv_core3.so
    optimized libopencv_highgui3.so
    optimized libopencv_imgcodecs3.so
    )


