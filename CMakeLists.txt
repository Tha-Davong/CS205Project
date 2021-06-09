cmake_minimum_required(VERSION 3.17)
project(CS205Project)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(Project ${OpenCV_INCLUDE_DIRS})

add_executable(CS205Project
        Project/main.cpp Project/complex.cpp
        Project/matrix.hpp Project/sparseMatrix.hpp Project/complex.h Project/templateUtil.h)

target_link_libraries(CS205Project ${OpenCV_LIBS})
