// src/cpp_inference/server.cpp
#include "crow_all.h"
#include "PlaygroundDetector.hpp" // 引入我们刚才写的类

// 注意：Docker 里的模型路径
const std::string MODEL_PATH = "/app/models/onnx/best.onnx";
// 也可以为了方便本地调试，写一段逻辑判断文件是否存在，如果不存在则使用本地路径

int main() {
    crow::SimpleApp app;

    // 1. 初始化模型 (只做一次！)
    // 为了兼容本地开发，我们可以检查一下路径
    std::string final_model_path = MODEL_PATH;
    std::ifstream f(final_model_path.c_str());
    if (!f.good()) {
        // 如果 Docker 路径不存在，尝试本地相对路径
        final_model_path = "../../../models/onnx/best.onnx"; 
    }
    
    // 实例化检测器
    PlaygroundDetector detector(final_model_path);

    // 2. 健康检查
    CROW_ROUTE(app, "/health")([](){ return "OK"; });

    // 3. 预测接口
    CROW_ROUTE(app, "/predict").methods(crow::HTTPMethod::Post)([&detector](const crow::request& req){
        auto json_body = crow::json::load(req.body);
        if (!json_body) return crow::response(400, "Invalid JSON");
        if (!json_body.has("image_path")) return crow::response(400, "Missing 'image_path'");

        std::string img_path = json_body["image_path"].s();
        
        // 调用真实的推理逻辑
        // 注意：这里会发生磁盘读取，生产环境通常传 Base64，但现在先这样
        std::vector<DetectionResult> results = detector.detect(img_path);

        // 构建 JSON 响应
        crow::json::wvalue resp;
        resp["status"] = "success";
        resp["count"] = results.size();
        
        // 只有 crow::json::wvalue::list 才能存数组
        crow::json::wvalue::list boxes_list;
        for (const auto& res : results) {
            crow::json::wvalue box_json;
            box_json["x"] = res.box.x;
            box_json["y"] = res.box.y;
            box_json["w"] = res.box.width;
            box_json["h"] = res.box.height;
            box_json["confidence"] = res.confidence;
            boxes_list.push_back(box_json);
        }
        resp["boxes"] = std::move(boxes_list);

        return crow::response(resp);
    });

    std::cout << ">>> Server running on 0.0.0.0:8080" << std::endl;
    app.port(8080).multithreaded().run();
}