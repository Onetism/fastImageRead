project(diffImage LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.0)

# find_package(CUDA REQUIRED)
# find_package(OpenCV REQUIRED)
# find_package(OpenMP REQUIRED)
# set(CMAKE_C_STANDARD   99)
# set(CMAKE_CXX_STANDARD 11)

add_subdirectory(src)

set(tgt main)
add_executable(${tgt} main.c++)
target_include_directories(${tgt} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src)
target_link_libraries(${tgt} imagesread ${OpenCV_LIBS})

