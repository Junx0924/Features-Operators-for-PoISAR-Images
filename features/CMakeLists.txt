PROJECT(features)
#SET(CMAKE_BUILD_TYPE Release)
CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

# Libraries
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

#find_package(Torch REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")


# To use GDAL, HDF5 libraries
# install GDAL and HDF5 from anaconda
link_directories(D:/Anaconda/Library/lib)
include_directories(D:/Anaconda/Library/include)
set(CMAKE_MSVCIDE_RUN_PATH  D:/Anaconda/Library/bin)
add_definitions(-DH5_BUILT_AS_DYNAMIC_LIB -D_HDF5USEDLL_)


add_subdirectory(src/MPEG-7)

add_library(feature
src/cv_hdf5.hpp
src/cv_hdf5.cpp
src/glcm.cpp
src/glcm.hpp
src/elbp.hpp
src/elbp.cpp
#src/sen12ms.cpp
#src/sen12ms.hpp
src/Geotiff.hpp
src/cvFeatures.cpp
src/cvFeatures.hpp
src/ober.hpp
src/ober.cpp
src/mp.cpp
src/mp.hpp
src/sarFeatures.hpp
src/sarFeatures.cpp
src/Utils.cpp
src/Utils.h
#src/torchDataSet.cpp
src/specklefilter.hpp
src/specklefilter.cpp
src/KNN.hpp
src/KNN.cpp
)

add_executable(main
src/main.cpp
)

set_property(TARGET main PROPERTY CXX_STANDARD 17)
set_property(TARGET feature PROPERTY CXX_STANDARD 17)


# libs to link with
target_link_libraries(feature ${OpenCV_LIBS}  gdal_i hdf5 MPEG-7)
target_link_libraries(main feature )


# The following code block is suggested to be used on Windows.
# if (MSVC)
#   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET main
#                     POST_BUILD
#                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                      ${TORCH_DLLS}
#                      $<TARGET_FILE_DIR:main>)
# endif (MSVC)
 