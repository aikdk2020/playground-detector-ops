# 🛰️ Playground Detection Ops

[![C++ Inference CI/CD](https://github.com/aikdk2020/playground-detector-ops/actions/workflows/ci.yml/badge.svg)](https://github.com/aikdk2020/playground-detector-ops/actions/workflows/ci.yml)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue?logo=docker)
![C++](https://img.shields.io/badge/C++-17-blue?logo=c%2B%2B)

基于 YOLO11 和 ONNX Runtime 的高性能遥感图像目标检测系统。包含完整的 C++ 推理引擎与 Docker/GitHub Actions 自动化部署流程。

## ✨ 核心特性
- **高性能**：使用 C++ 编写，基于 ONNX Runtime CPU 推理。
- **轻量化**：通过 Docker 容器化，无复杂的 Python 依赖。
- **自动化**：集成 GitHub Actions CI/CD，自动测试与构建。

## 🚀 快速开始

### 方式一：Docker 运行 (推荐)
无需配置环境，直接运行：
```bash
# 1. 拉取镜像 (或者自己 build)
docker build --network=host -t playground-detector:latest .

# 2. 准备图片并运行
# 确保当前目录下有 data/test_images 文件夹
docker run --rm -v $(pwd)/data:/app/data playground-detector:latest
# 结果将保存在 data/inference_results 中
```
### 方式二：源码编译
```bash
mkdir -p src/cpp_inference/build
cd src/cpp_inference/build
cmake ..
make
./detector_app
```
## 📂目录结构
- **models/**: ONNX 模型文件
- **src/inference/**: Python 原型代码
- **src/cpp_inference/**: C++ 核心推理代码
- **.github/workflows/**: CI/CD 自动化脚本
- **Dockerfile**: 容器化构建文件
