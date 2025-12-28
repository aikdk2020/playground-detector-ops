#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <iostream>

// 定义一个结构体来存储检测结果
struct DetectionResult {
    cv::Rect box;
    float confidence;
};

class PlaygroundDetector {
private:
    Ort::Env env{nullptr};
    Ort::Session session{nullptr};
    Ort::AllocatorWithDefaultOptions allocator;
    
    // 模型参数
    const float CONF_THRESHOLD = 0.5f;
    const float IOU_THRESHOLD = 0.45f;
    const int INPUT_W = 640;
    const int INPUT_H = 640;
    
    // 动态获取的节点名称
    std::string input_name_str;
    std::string output_name_str;
    const char* input_name = nullptr;
    const char* output_name = nullptr;

public:
    // 构造函数：加载模型
    PlaygroundDetector(const std::string& model_path) {
        std::cout << ">>> Initializing Detector with model: " << model_path << std::endl;
        
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "PlaygroundService");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        
        session = Ort::Session(env, model_path.c_str(), session_options);

        // 获取节点名称 (这对 ONNX Runtime 非常重要)
        // 注意：这里必须深拷贝字符串，因为 GetInputNameAllocated 返回的智能指针释放后指针会失效
        auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
        auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
        
        input_name_str = input_name_ptr.get();
        output_name_str = output_name_ptr.get();
        
        input_name = input_name_str.c_str();
        output_name = output_name_str.c_str();
    }

    // 核心推理函数
    std::vector<DetectionResult> detect(const std::string& image_path) {
        std::vector<DetectionResult> results;

        // 1. 读取图片
        cv::Mat img = cv::imread(image_path);
        if (img.empty()) {
            std::cerr << "Error: Could not read image: " << image_path << std::endl;
            return results;
        }

        int original_w = img.cols;
        int original_h = img.rows;

        // 2. 预处理
        cv::Mat blob;
        cv::dnn::blobFromImage(img, blob, 1.0/255.0, cv::Size(INPUT_W, INPUT_H), cv::Scalar(), true, false);

        size_t input_tensor_size = 1 * 3 * INPUT_W * INPUT_H;
        std::vector<int64_t> input_node_dims = {1, 3, INPUT_H, INPUT_W};
        
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, blob.ptr<float>(), input_tensor_size, input_node_dims.data(), input_node_dims.size()
        );

        // 3. 推理
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1
        );

        // 4. 后处理
        float* floatarr = output_tensors[0].GetTensorMutableData<float>();
        
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        
        int rows = 8400; // YOLO11 output anchors
        float x_factor = (float)original_w / INPUT_W;
        float y_factor = (float)original_h / INPUT_H;

        for (int i = 0; i < rows; ++i) {
            float confidence = floatarr[4 * rows + i];
            if (confidence >= CONF_THRESHOLD) {
                float cx = floatarr[0 * rows + i];
                float cy = floatarr[1 * rows + i];
                float w  = floatarr[2 * rows + i];
                float h  = floatarr[3 * rows + i];

                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(confidence);
            }
        }

        // 5. NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD, indices);

        for (int idx : indices) {
            results.push_back({boxes[idx], confidences[idx]});
        }
        
        return results;
    }
};