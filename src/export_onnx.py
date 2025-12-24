# src/export_onnx.py
from ultralytics import YOLO

# 加载你的模型
model = YOLO("models/raw/best.pt")

# 导出为 ONNX 格式
# opset=12 是一个兼容性很好的版本
# dynamic=True 允许输入不同尺寸的图片（对遥感切片很重要）
success = model.export(format="onnx", opset=12, dynamic=True)

if success:
    print("Export success! Moving file...")
    import shutil
    import os
    # 假设导出文件在原路径，将其移动到规范目录
    source = "models/raw/best.onnx"
    destination = "models/onnx/best.onnx"
    if os.path.exists(source):
        os.rename(source, destination)
        print(f"Model saved to {destination}")