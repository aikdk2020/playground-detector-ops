cmake_minimum_required(VERSION 3.10)
project(PlaygroundDetectorCPP)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)

# 1. 查找 OpenCV
find_package(OpenCV REQUIRED)

# 2. 设置 ONNX Runtime 路径 (指向我们刚才在 lib 下解压的目录)
# ${CMAKE_SOURCE_DIR} 代表当前 CMakeLists.txt 所在的目录
set(ORT_HOME "${CMAKE_SOURCE_DIR}/../../lib/onnxruntime")

# 检查路径是否正确
message(STATUS "ONNX Runtime Path: ${ORT_HOME}")

# 包含头文件
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ORT_HOME}/include)

# 链接库目录
link_directories(${ORT_HOME}/lib)

# 3. 创建可执行文件
add_executable(detector_app main.cpp)

# 4. 链接库文件
target_link_libraries(detector_app 
    ${OpenCV_LIBS} 
    onnxruntime  # 链接 libonnxruntime.so
)

# 5. 这一步是为了运行时能找到动态库 (RPATH设置)
# 否则运行程序时会报错 "error while loading shared libraries"
set_target_properties(detector_app PROPERTIES 
    BUILD_RPATH "${ORT_HOME}/lib"
    INSTALL_RPATH "${ORT_HOME}/lib"
)