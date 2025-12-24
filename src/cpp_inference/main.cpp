#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>

// 配置参数
const std::string MODEL_PATH = "/app/models/onnx/best.onnx";
const std::string TEST_IMAGE_PATH = "/app/data/test_images/playground_209.jpg";
const std::string OUTPUT_IMAGE_PATH = "/app/data/inference_results/playground_209_docker.jpg";
const float CONF_THRESHOLD = 0.5f;
const float IOU_THRESHOLD = 0.45f;
const int INPUT_W = 640;
const int INPUT_H = 640;

int main() {
    // 1. 初始化 ONNX Runtime 环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "PlaygroundDetector");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    
    // 加载模型
    std::cout << " Loading model..." << std::endl;
    Ort::Session session(env, MODEL_PATH.c_str(), session_options);

    // 获取输入输出节点名称 (为了通用性，我们动态获取)
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    const char* input_name = input_name_ptr.get();
    const char* output_name = output_name_ptr.get();

    // 2. 图像预处理 (Pre-processing)
    std::cout << " Reading image: " << TEST_IMAGE_PATH << std::endl;
    cv::Mat img = cv::imread(TEST_IMAGE_PATH);
    if (img.empty()) {
        std::cerr << "Error: Could not read image!" << std::endl;
        return -1;
    }

    // 记录原始尺寸用于后续坐标还原
    int original_w = img.cols;
    int original_h = img.rows;

    // 使用 OpenCV DNN 模块进行预处理 (Resize + Normalize + CHW 转换)
    // 这里的参数必须和 Python 保持一致：1/255.0, swapRB=True
    cv::Mat blob;
    cv::dnn::blobFromImage(img, blob, 1.0/255.0, cv::Size(INPUT_W, INPUT_H), cv::Scalar(), true, false);

    // 3. 准备输入 Tensor
    // blobFromImage 返回的数据已经是连续的内存块，我们可以直接用指针
    size_t input_tensor_size = 1 * 3 * INPUT_W * INPUT_H;
    std::vector<int64_t> input_node_dims = {1, 3, INPUT_H, INPUT_W};
    
    // 创建内存信息
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // 创建输入 Tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        blob.ptr<float>(), // 获取 OpenCV blob 的原始数据指针
        input_tensor_size, 
        input_node_dims.data(), 
        input_node_dims.size()
    );

    // 4. 执行推理 (Inference)
    std::cout << " Running inference..." << std::endl;
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, 
        &input_name, &input_tensor, 1, 
        &output_name, 1
    );

    // 5. 后处理 (Post-processing) - 这里的逻辑最“C++”
    float* floatarr = output_tensors[0].GetTensorMutableData<float>();
    
    // 输出形状是 [1, 5, 8400] -> Batch=1, Channels=5, Anchors=8400
    // 内存布局是：[cx0...cx8399, cy0...cy8399, w0...w8399, h0...h8399, conf0...conf8399]
    // Python 里的转置 (.T) 在这里需要我们要跳跃访问
    
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;

    int dimensions = 5; // cx, cy, w, h, conf
    int rows = 8400;    // 8400 个预测框

    float x_factor = (float)original_w / INPUT_W;
    float y_factor = (float)original_h / INPUT_H;

    for (int i = 0; i < rows; ++i) {
        // 关键点：如何从 [5, 8400] 的扁平数组里取值
        // 第 i 个框的置信度在第 4 行 (0-indexed)
        float confidence = floatarr[4 * rows + i];

        if (confidence >= CONF_THRESHOLD) {
            // 获取坐标
            float cx = floatarr[0 * rows + i];
            float cy = floatarr[1 * rows + i];
            float w  = floatarr[2 * rows + i];
            float h  = floatarr[3 * rows + i];

            // 还原到原图坐标
            int left = int((cx - 0.5 * w) * x_factor);
            int top = int((cy - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(confidence);
        }
    }

    std::cout << " Candidates after threshold: " << boxes.size() << std::endl;

    // 6. NMS (非极大值抑制)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD, indices);

    std::cout << " Final detections: " << indices.size() << std::endl;

    // 7. 绘图与保存
    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        
        // 画框 (绿色)
        cv::rectangle(img, box, cv::Scalar(0, 255, 0), 2);

        // 写标签
        std::string label = std::to_string(confidences[idx]).substr(0, 4);
        cv::putText(img, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }

    cv::imwrite(OUTPUT_IMAGE_PATH, img);
    std::cout << " Saved result to: " << OUTPUT_IMAGE_PATH << std::endl;

    return 0;
}