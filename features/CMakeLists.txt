PROJECT(features)
SET(CMAKE_BUILD_TYPE Release)
CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )

cmake_policy(SET CMP0015 NEW)
# Libraries
# find_package(OpenCV REQUIRED PATHS ${CMAKE_CURRENT_LIST_DIR}/lib/opencv)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(features
 main.cpp
)

add_subdirectory(MPEG-7)
add_subdirectory(ELBP)
add_subdirectory(GLCM)
#add_subdirectory(PolSAR)



# libs to link with
target_link_libraries(features  ${OpenCV_LIBS} MPEG-7 ELBP GLCM  )