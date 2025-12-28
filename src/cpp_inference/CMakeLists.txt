cmake_minimum_required(VERSION 3.10)
project(PlaygroundDetectorCPP)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 设置 ONNX Runtime 路径 (指向我们刚才在 lib 下解压的目录)
# ${CMAKE_SOURCE_DIR} 代表当前 CMakeLists.txt 所在的目录
set(ORT_HOME "${CMAKE_SOURCE_DIR}/../../lib/onnxruntime")
set(CROW_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../../lib/crow/include")

# 检查路径是否正确
message(STATUS "ONNX Runtime Path: ${ORT_HOME}")

# 包含头文件
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ORT_HOME}/include)
include_directories(${CROW_INCLUDE_DIR})

# 链接库目录
link_directories(${ORT_HOME}/lib)

# 创建可执行文件
add_executable(detector_app main.cpp)

# 链接库文件
target_link_libraries(detector_app 
    ${OpenCV_LIBS} 
    onnxruntime  # 链接 libonnxruntime.so
)

# 这一步是为了运行时能找到动态库 (RPATH设置)
# 否则运行程序时会报错 "error while loading shared libraries"
set_target_properties(detector_app PROPERTIES 
    BUILD_RPATH "${ORT_HOME}/lib"
    INSTALL_RPATH "${ORT_HOME}/lib"
)

# --- 新增的 Server 程序 ---
add_executable(detector_server server.cpp) # 新增
target_link_libraries(detector_server ${OpenCV_LIBS} onnxruntime pthread) # Crow需要pthread
set_target_properties(detector_server PROPERTIES 
    BUILD_RPATH "${ORT_HOME}/lib"
    INSTALL_RPATH "${ORT_HOME}/lib"
)