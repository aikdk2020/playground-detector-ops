# 🛰️ Playground Detection Ops

[![C++ Inference CI/CD](https://github.com/aikdk2020/playground-detector-ops/actions/workflows/ci.yml/badge.svg)](https://github.com/aikdk2020/playground-detector-ops/actions/workflows/ci.yml)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue?logo=docker)
![Python](https://img.shields.io/badge/Python-14354C.svg?logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C++-17-blue?logo=c%2B%2B)

企业级的高性能遥感图像检测系统, 展示 MLOps 全链路 的实践案例。它实现了从算法模型管理、C++ 高性能微服务封装，到 Kubernetes 云原生部署与自动弹性伸缩（HPA）的完整闭环。

## ✨ 核心特性
**⚡ 高性能 (High Performance)**:
- 后端核心采用 C++17 重构，基于 Crow 异步 Web 框架。
- 集成 ONNX Runtime C++ API 进行推理加速，相比 Python 原型延迟降低。
- 实现 RAII 资源管理，杜绝内存泄漏。

**☁️ 云原生架构 (Cloud-Native)**:
- 完全 Docker 容器化.
- 基于 Kubernetes (Minikube) 编排，配置 Deployment 实现双副本高可用
- HPA 自动伸缩: 当 CPU 利用率超过 50% 时，Pod 自动从 2 个扩容至 5 个以应对突发流量。

**🛠️ 工程化治理 (MLOps):**
- CI/CD: 集成 GitHub Actions，自动完成 C++ 编译、Docker 构建、Python 语法检查及集成测试。
- 数据治理: 使用 DVC (Data Version Control) + 阿里云 OSS 管理大模型文件，实现代码与数据的解耦。

**📊 全栈交互:**
- 提供基于 Streamlit 的可视化 Web 前端。
- 实现了二进制流透传协议，解决 Web 端图片编解码导致的模型精度损失问题。

## 🚀 快速开始
### 前置要求 (Prerequisites):
- OS: Ubuntu 20.04/22.04/24.04 (推荐)
- Docker & Minikube 已安装
- Python 3.8+ & pip
### 1. 克隆项目与环境准备
```bash
git clone https://github.com/aikdk2020/playground-detector-ops.git
cd playground_detection_ops
```
### 2. 准备模型文件
你需要下载预训练好的 YOLO11 模型文件。
请前往 [Releases 页面](https://github.com/aikdk2020/playground-detector-ops/releases) 下载 `best.onnx` 文件，并将其放入 `models/onnx/` 目录中。

### 3. 启动 Kubernetes 集群与部署
#### 3.1: 启动 Minikube
```bash
minikube start
# 开启 Metrics Server (为了 HPA 自动伸缩)
minikube addons enable metrics-server
```
#### 3.2: 挂载数据卷K8s 需要读取宿主机的模型和数据。请打开一个新的终端窗口执行以下命令，并保持该窗口开启：
```bash
# 在新终端中执行
minikube mount $(pwd)/data:/data/playground_project
```
#### 3.3: 构建镜像并部署 回到原来的终端：
```bash
# 1. 在本地构建 Docker 镜像
docker build --network=host -t playground-service:latest .

# 2. 将镜像加载到 Minikube 内部 (耗时可能较长)
minikube image load playground-service:latest

# 3. 应用 K8s 配置
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# 4. 等待 Pod 启动 (状态变为 Running 即成功)
kubectl get pods -w
```
### 4. 启动前端可视化演示
```bash
# 安装依赖
pip install streamlit requests pillow numpy

# 获取 Minikube IP 
minikube ip

# 启动 Web App
streamlit run frontend/app.py
```
此时，浏览器会自动打开 http://localhost:8501。

上传 data/test_images/ 下的图片。

点击 "🚀 开始检测"。

## 📂项目结构
```Plaintext
.
├── .dvc                 # DVC 配置
├── .github/workflows/   # GitHub Actions CI 配置
├── data/                # 测试数据集 (由 DVC 挂载)
├── frontend/            # Streamlit 前端代码
│   └── app.py
├── k8s/                 # Kubernetes 资源清单 (Deployment, Service, HPA)
├── models/              # 模型文件 (DVC 追踪)
├── src/
│   └── cpp_inference/   # C++ 核心推理源码
│       ├── PlaygroundDetector.hpp  # 推理类封装
│       ├── server.cpp              # Crow 微服务入口
│       └── CMakeLists.txt
├── Dockerfile           # 多阶段构建脚本
└── README.md
```
## 🧪 验证自动伸缩
如果你想验证系统的高并发抗压能力：
打开一个终端，运行压测循环：
(建议开 3-4 个终端同时运行此命令)
```bash
export IP=$(minikube ip)
while true; do curl -s -X POST -H "Content-Type: application/json" -d '{"image_path": "/app/data/test_images/playground_209.jpg"}' http://$IP:30008/predict > /dev/null; done
```

观察 HPA 状态：
```bash
kubectl get hpa -w
```
你会看到 TARGETS 飙升超过 50%，随后 REPLICAS 会自动从 2 增加到 5。

## 🛠️ 技术栈详情
- Languages: C++ 17, Python 3.9

- Web Framework: Crow (C++), Streamlit (Python)

- Deep Learning: YOLO11, ONNX Runtime

- Containerization: Docker

- Orchestration: Kubernetes (Minikube)

- Version Control: Git, DVC (Data Version Control)

- Cloud Storage: Aliyun OSS