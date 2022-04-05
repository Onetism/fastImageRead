cmake_minimum_required(VERSION 3.11.0)
project(imreadfast)

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)


# Update libklb git submodule,
# adapted from https://cliutils.gitlab.io/modern-cmake/chapters/projects/submodule.html
find_package(Git REQUIRED)
message(STATUS "Submodule update")
execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}")
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/libimagesread/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()


include_directories(${INCLUDE_DIRECTORIES} ${NumPy_INCLUDE_DIR} libimagesread/src)

add_subdirectory(pyimread)

add_subdirectory(libimagesread)
# set(BINDINGS_JAVA       OFF)
# set(build_tests         OFF)
# set(SKIP_INSTALL_ALL    TRUE)
# set(SKIP_INSTALL_EXPORT TRUE)
# set(SKIP_TESTS          TRUE)
set_target_properties(imagesread PROPERTIES EXCLUDE_FROM_ALL TRUE)









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

