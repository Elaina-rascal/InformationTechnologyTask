cmake_minimum_required(VERSION 3.22)

set(PROJECT_NAME "task")
project(${PROJECT_NAME} LANGUAGES CXX )
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
#设置项目名称

set(exe1_name "encode")
# find_package(Threads REQUIRED)

#新建可执行文件
add_executable(${exe1_name} encode.cpp)
target_include_directories(${exe1_name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
#添加链接库
target_link_libraries(${exe1_name} )