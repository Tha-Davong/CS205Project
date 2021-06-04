cmake_minimum_required(VERSION 3.17)
project(CS205Project)

set(CMAKE_CXX_STANDARD 14)

include_directories(Project)

add_executable(CS205Project
        Project/main.cpp
        Project/matrix.hpp Project/sparseMatrix.hpp Project/complex.h Project/complex.cpp Project/sparseMatrix.cpp)
