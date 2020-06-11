PROJECT(features)
SET(CMAKE_BUILD_TYPE Release)
CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )

cmake_policy(SET CMP0015 NEW)
# Libraries
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#find_package(Torch REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")


# To use GDAL libraries:
# Install GDAL in conda,
# conda install -c conda-forge gdal
# Add conda lib and include path
link_directories(D:/Anaconda/Library/lib)
include_directories(D:/Anaconda/Library/include)
set(CMAKE_MSVCIDE_RUN_PATH  D:/Anaconda/Library/bin)

# To use Eigen3.3.3
include_directories("D:/Program Files/Eigen")

add_subdirectory(src/MPEG-7)

add_library(feature
src/glcm.cpp
src/glcm.hpp
src/elbp.hpp
src/elbp.cpp
src/sen12ms.cpp
src/sen12ms.hpp
src/Geotiff.cpp
src/cvFeatures.cpp
src/cvFeatures.hpp
src/ober.hpp
src/ober.cpp
src/mp.cpp
src/mp.hpp
src/sarFeatures.hpp
src/sarFeatures.cpp
)

add_executable(main
src/main.cpp
)

set_property(TARGET main PROPERTY CXX_STANDARD 17)
set_property(TARGET feature PROPERTY CXX_STANDARD 17)


# libs to link with
#target_link_libraries(feature ${OpenCV_LIBS} ${TORCH_LIBRARIES} gdal_i MPEG-7)
target_link_libraries(feature ${OpenCV_LIBS}  MPEG-7 gdal_i)
target_link_libraries(main feature )


# The following code block is suggested to be used on Windows.
#if (MSVC)
#  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#  add_custom_command(TARGET main
#                     POST_BUILD
#                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                     ${TORCH_DLLS}
#                     $<TARGET_FILE_DIR:main>)
#endif (MSVC)
 