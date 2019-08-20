cmake_minimum_required(VERSION 3.5)
SET(PROJ_NAME spix)
PROJECT(${PROJ_NAME})

ADD_DEFINITIONS(-std=c++11)

INCLUDE_DIRECTORIES(
    /usr/local/include
    /usr/local/include/opencv4
    )

LINK_DIRECTORIES(
    /usr/local/lib/
    )

FIND_PACKAGE(OpenCV 4.1.0 REQUIRED )

ADD_EXECUTABLE(
    ${PROJ_NAME}
    main.cpp
    MBS.cpp
    )

TARGET_LINK_LIBRARIES(${PROJ_NAME}
    optimized libopencv_imgproc.so
    optimized libopencv_core.so
    optimized libopencv_highgui.so
    optimized libopencv_imgcodecs.so
    )


