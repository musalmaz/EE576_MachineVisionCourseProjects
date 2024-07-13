# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.0)

# Project's name
project(sinav)

# Set the C++ standard you are using (e.g., C++11)
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV package on your system
find_package(OpenCV REQUIRED)

# Add include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Specify the executable target
add_executable(main main.cpp)

# Link your application with OpenCV libraries
target_link_libraries(main ${OpenCV_LIBS})