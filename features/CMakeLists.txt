PROJECT(features)
SET(CMAKE_BUILD_TYPE Release)
CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )

cmake_policy(SET CMP0015 NEW)
# Libraries
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# To use GDAL libraries:
# Install GDAL in conda,
# conda install -c conda-forge gdal
# Add conda lib and include path
link_directories(D:/Anaconda/Library/lib)
include_directories(D:/Anaconda/Library/include)
set(CMAKE_MSVCIDE_RUN_PATH  D:/Anaconda/Library/bin)

add_subdirectory(src)
add_subdirectory(data)

add_library(feature
src/glcm.cpp
src/glcm.hpp
src/elbp.cpp
src/elbp.hpp
src/sen12ms.cpp
src/sen12ms.hpp
src/Geotiff.cpp
src/torchDataSet.cpp
)


add_executable(main
src/main.cpp
)

set_property(TARGET main PROPERTY CXX_STANDARD 14)

# libs to link with
target_link_libraries(feature ${OpenCV_LIBS} ${TORCH_LIBRARIES} gdal_i)
target_link_libraries(main   feature )



# The following code block is suggested to be used on Windows.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET main
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:main>)
endif (MSVC)
 