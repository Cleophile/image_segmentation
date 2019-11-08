cmake_minimum_required(VERSION 3.10)
SET(PROJ_NAME mbs)
PROJECT(${PROJ_NAME})

ADD_DEFINITIONS(-std=c++11)
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

INCLUDE_DIRECTORIES(
	/usr/local/include
	/usr/local/include/opencv4
	)

LINK_DIRECTORIES(
	/usr/local/lib/
	)

FIND_PACKAGE(OpenCV 4.1.1 REQUIRED)

ADD_EXECUTABLE(
	${PROJ_NAME}
	main.cpp
	MBS.cpp
	)

TARGET_LINK_LIBRARIES(${PROJ_NAME}
	optimized libopencv_imgproc.4.1.1.dylib
	optimized libopencv_core.4.1.1.dylib
	optimized libopencv_highgui.4.1.1.dylib
	optimized libopencv_imgcodecs.4.1.1.dylib
	)

