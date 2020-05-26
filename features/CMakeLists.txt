PROJECT(features)
SET(CMAKE_BUILD_TYPE Release)
CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )

cmake_policy(SET CMP0015 NEW)
# Libraries
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# To use GDAL libraries:
# Install GDAL in conda,
# conda install -c conda-forge gdal
# Add conda lib and include path
link_directories(D:/Anaconda/Library/lib)
include_directories(D:/Anaconda/Library/include)
set(CMAKE_MSVCIDE_RUN_PATH  D:/Anaconda/Library/bin)

add_executable(features
 main.cpp
)

add_subdirectory(MPEG-7)
add_subdirectory(ELBP)
add_subdirectory(GLCM)
add_subdirectory(PolSAR)



# libs to link with
target_link_libraries(features  ${OpenCV_LIBS} MPEG-7 ELBP GLCM  PolSAR)