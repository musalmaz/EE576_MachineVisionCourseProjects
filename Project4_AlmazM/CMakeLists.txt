# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.0)

# Project's name
project(Project4_AlmazM)

# Set the C++ standard you are using (e.g., C++11)
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV package on your system
set(OPENCV_ENABLE_NONFREE)
# set(OpenCV_DIR "/home/musa/opencv_build/opencv/build")
find_package(OpenCV REQUIRED)

find_package(PythonLibs 3.7 REQUIRED)  # Replace '2.7' with your Python version
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(PCL 1.8 REQUIRED COMPONENTS common io)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Add include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Specify the executable target
add_executable(main main.cpp project4.hpp project4.cpp)

# Link your application with OpenCV libraries
target_link_libraries(main ${OpenCV_LIBS} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES})