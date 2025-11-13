cmake_minimum_required(VERSION 3.22)

set( CMAKE_CXX_STANDARD 20)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
#设置项目名称
set(PROJECT_NAME "task")
set(exe1_name "encode")
project(${PROJECT_NAME} VERSION 1.0.0)
#新建可执行文件
add_executable(${exe1_name} encode.cpp)
target_include_directories(${exe1_name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
#添加链接库
target_link_libraries(${exe1_name} PRIVATE DataFrame )