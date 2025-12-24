import onnxruntime as ort
import numpy as np
import cv2
import os
import glob
from tqdm import tqdm  # 引入进度条库

class PlaygroundDetector:
    def __init__(self, model_path, conf_thres=0.5, iou_thres=0.45):
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
            
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = (640, 640)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def preprocess(self, img):
        self.img_h, self.img_w = img.shape[:2]
        img_resized = cv2.resize(img, self.input_shape)
        blob = cv2.dnn.blobFromImage(img_resized, 1/255.0, self.input_shape, swapRB=True, crop=False)
        return blob

    def postprocess(self, outputs, original_img):
        predictions = np.squeeze(outputs).T 
        scores = predictions[:, 4]
        keep_idxs = scores > self.conf_thres
        predictions = predictions[keep_idxs]
        scores = scores[keep_idxs]

        if len(predictions) == 0:
            return original_img

        x_factor = self.img_w / self.input_shape[0]
        y_factor = self.img_h / self.input_shape[1]

        boxes = []
        for row in predictions:
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            left = int((cx - w/2) * x_factor)
            top = int((cy - h/2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, scores.tolist(), self.conf_thres, self.iou_thres)

        if len(indices) > 0:
            # 使用你修复的 flatten() 方法
            for i in indices.flatten():
                box = boxes[i]
                left, top, width, height = box[0], box[1], box[2], box[3]
                
                # 画框 (绿色)
                cv2.rectangle(original_img, (left, top), (left + width, top + height), (0, 255, 0), 2)
                
                # 标签背景条 (增加可读性)
                label = f"{scores[i]:.2f}"
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(original_img, (left, top - label_size[1] - 10), (left + label_size[0], top), (0, 255, 0), -1)
                
                # 写字 (黑色)
                cv2.putText(original_img, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        return original_img

    def process_batch(self, input_dir, output_dir):
        # 1. 准备输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # 2. 获取所有图片文件 (支持 jpg, jpeg, png)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return

        print(f"Starting inference on {len(image_files)} images...")

        # 3. 批量循环处理 (带进度条)
        for img_path in tqdm(image_files, desc="Processing"):
            try:
                # 读取图片
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue

                # 推理
                blob = self.preprocess(img)
                outputs = self.session.run(None, {self.input_name: blob})
                result_img = self.postprocess(outputs[0], img)

                # 保存结果
                filename = os.path.basename(img_path)
                save_path = os.path.join(output_dir, filename)
                cv2.imwrite(save_path, result_img)
            
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        print(f"\nAll done! Results saved to: {output_dir}")

if __name__ == "__main__":
    detector = PlaygroundDetector("models/onnx/best.onnx")
    
    # 定义输入输出目录
    input_folder = "data/test_images"
    output_folder = "data/inference_results" # 结果单独放这里
    
    detector.process_batch(input_folder, output_folder)