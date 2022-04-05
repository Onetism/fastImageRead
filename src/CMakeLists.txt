cmake_policy(SET CMP0104 OLD)

set(CUDA_NVCC_FLAGS  ${CUDA_NVCC_FLAGS};-Xcompiler -fPIC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)  

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB SRCS *.c++ *.cu)
file(GLOB HDRS *.h)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/imagesread.h DESTINATION ${PROJECT_SOURCE_DIR}/include)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

set(tgt imagesread)
add_library(${tgt} SHARED ${SRCS} ${HDRS})
target_include_directories(${tgt} PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(${tgt} ${OpenCV_LIBS} ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)
# set_target_properties(${tgt} PROPERTIES CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_LIST_DIR}/build/lib)

set(tgt imagesread_static)
add_library(${tgt} STATIC ${SRCS} ${HDRS})
target_include_directories(${tgt} PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(${tgt} ${OpenCV_LIBS} ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)
# set_target_properties(${tgt} PROPERTIES CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_LIST_DIR}/build/lib)
# set(tgt diffimage)
# add_library(${tgt} SHARED diffimage.c++ diffimage.h)
# # add_executable(${tgt} diffimage.c++ diffimage.h)
# target_link_libraries (${tgt} diffcuda ${OpenCV_LIBS} OpenMP::OpenMP_CXX ${CUDA_LIBRARIES})
# target_include_directories(${tgt} PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})



