
project(stereoVision)
# 在cmake脚本中，设置编译选项可以通过add_compile_options命令，也可以通过set命令修改CMAKE_CXX_FLAGS或CMAKE_C_FLAGS。
# add_compile_options(-std=c++11) # 这句话的意思是在make时让编译器（g++）自动添加C++11的性质 -> 设置编译选项

option(USE_UBUNTU_18 "Set to ON if you are using Ubuntu 18.04" ON)
find_package(Pangolin REQUIRED)
find_package(OpenCV 3.4 REQUIRED)
if(USE_UBUNTU_18)
    message("You are using Ubuntu 18.04, fmt::fmt will be linked")
    find_package(fmt REQUIRED)
    set(FMT_LIBRARIES fmt::fmt)
endif()
include_directories(${Pangolin_INCLUDE_DIRS})
add_executable(stereoVision stereoVision.cpp)
target_link_libraries(stereoVision ${FMT_LIBRARIES})
target_link_libraries(stereoVision ${Pangolin_LIBRARIES})
cmake_minimum_required(VERSION 3.0)
set(CMAKE_BUILD_TYPE "Release")
set(Sophus_LIBRARIES  "/usr/local/lib/libSophus.so")
find_package(Sophus REQUIRED)
include_directories( ${Sophus_INCLUDE_DIRS} )
target_link_libraries(stereoVision ${Sophus_LIBRARIES})
target_link_libraries(stereoVision ${OpenCV_LIBS} ${Pangolin_LIBRARIES})


#include_directories( ${Sophus_INCLUDE_DIRS} )
#target_link_libraries(jointMap ${Sophus_LIBRARIES})
#target_link_libraries(jointMap ${OpenCV_LIBS} ${Pangolin_LIBRARIES})

