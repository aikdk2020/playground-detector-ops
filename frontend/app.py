import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import json
import os

# --- 页面配置 ---
st.set_page_config(
    page_title="遥感图像操场检测系统",
    page_icon="🛰️",
    layout="wide"
)

# --- 侧边栏配置 ---
st.sidebar.title("⚙️ 系统配置")
st.sidebar.markdown("基于 Kubernetes + C++ Microservice")

# 自动获取 Minikube IP (如果环境变量有的话)，否则默认本地
default_ip = os.environ.get("MINIKUBE_IP", "192.168.58.2") 
api_url = st.sidebar.text_input("后端 API 地址", f"http://{default_ip}:30008/predict")

confidence_threshold = st.sidebar.slider("可视化置信度阈值", 0.0, 1.0, 0.5, 0.05)

# --- 主页面 ---
st.title("🛰️ Remote Sensing Playground Detection")
st.markdown("### Enterprise-Grade AI Inference System")

# 1. 文件上传
uploaded_file = st.file_uploader("上传一张遥感图片 (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 展示原始图片
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("原始图片")
        st.image(image, use_container_width=True)

    # 2. 发起推理请求
    if st.button("🚀 开始检测 (Start Inference)"):
        with st.spinner("正在请求 Kubernetes 集群进行推理..."):
            try:
                # 为了简化演示，我们这里通过一种 Hack 的方式
                # 因为后端目前设计是读本地路径，但 Streamlit 上传的是内存流
                # 正常做法是后端支持文件流上传。
                # 这里的变通方案：我们把图保存到之前挂载的 data 目录，让后端去读
                
                # 获取上传文件的原始扩展名 (如 .jpg, .png)
                file_ext = os.path.splitext(uploaded_file.name)[1]
        
                # 构造保存路径 (文件名保持简单，但保留原始后缀)
                save_filename = f"temp_upload{file_ext}"
                # 使用 os.path.join 确保跨平台路径正确
                save_path = os.path.join(os.getcwd(), "data", save_filename)

                # 写入二进制流
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 构造给 Docker 内部读取的路径
                docker_internal_path = f"/app/data/{save_filename}"
                
                payload = {"image_path": docker_internal_path}
                
                # 发送请求
                response = requests.post(api_url, json=payload, timeout=5)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # 3. 绘制结果
                    draw = ImageDraw.Draw(image)
                    # 尝试加载字体，如果失败就用默认
                    try:
                        font = ImageFont.truetype("DejaVuSans.ttf", 20)
                    except:
                        font = ImageFont.load_default()
                    
                    count = 0
                    for box in result.get("boxes", []):
                        conf = box.get("confidence", 0)
                        if conf < confidence_threshold:
                            continue
                            
                        count += 1
                        x = box["x"]
                        y = box["y"]
                        w = box["w"]
                        h = box["h"]
                        
                        # 画框 (PIL 接收 [x0, y0, x1, y1])
                        # 假设后端返回的是中心点坐标 (x,y) 和宽高 (w,h) -> 根据你 C++ 代码确认
                        # 你的 C++ 代码里：
                        # result["boxes"][0]["x"] = 100; (如果是左上角)
                        # 请检查你的 C++ 逻辑。如果是 OpenCV Rect，通常是左上角 x,y
                        
                        # 假设是 左上角 X,Y
                        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
                        draw.text((x, y - 25), f"{conf:.2f}", fill="red", font=font)
                    
                    with col2:
                        st.subheader(f"检测结果 (发现 {count} 个目标)")
                        st.image(image, use_container_width=True)
                        
                    # 展示 JSON 数据 (给面试官看 Raw Data)
                    with st.expander("查看原始 JSON 响应 (Debug Info)"):
                        st.json(result)
                        
                    st.success(f"✅ 推理成功！耗时: {response.elapsed.total_seconds() * 1000:.2f} ms")
                
                else:
                    st.error(f"❌ 请求失败: {response.status_code} - {response.text}")
                    
            except Exception as e:
                st.error(f"❌ 连接错误: {str(e)}")
                st.info("提示：请检查 Minikube IP 是否正确，以及 K8s Service 是否存活。")