# 使用和宿主机一致的基础镜像，减少版本兼容问题
FROM ubuntu:24.04

# 1. 设置非交互模式，避免安装 apt 包时弹出时区选择框卡住
ENV DEBIAN_FRONTEND=noninteractive

# 2. 安装编译环境和 OpenCV 依赖
# build-essential: 编译器
# cmake: 构建工具
# libopencv-dev: OpenCV 库
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. 设置工作目录
WORKDIR /app

# 4. 复制依赖库 (ONNX Runtime)
# 将本地的 lib 目录复制到镜像中的 /app/lib
COPY lib /app/lib

# 5. 复制源代码和模型
COPY src /app/src
COPY models /app/models

# 6. 编译 C++ 项目
# 创建构建目录 -> 运行 CMake -> 运行 Make
WORKDIR /app/src/cpp_inference/build
RUN cmake .. && make

# 7. 设置环境变量
# 这一步至关重要！告诉 Linux 运行时去哪里找 libonnxruntime.so
ENV LD_LIBRARY_PATH="/app/lib/onnxruntime/lib"

# 8. 默认运行命令
CMD ["./detector_app"]