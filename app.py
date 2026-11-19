import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# 1. Page Config
st.set_page_config(page_title="Kidney Stone Detector", layout="wide")
st.title("🩺 Kidney Stone Detector (YOLOv8)")
st.write("Upload a CT scan image to detect kidney stones. Adjust the confidence slider to filter results.")

# 2. Load Model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# 3. Sidebar Slider (Required by the task)
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05
)

# 4. File Uploader
uploaded_file = st.file_uploader("Choose a CT scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)

    # Run YOLO inference
    results = model.predict(source=img_array, conf=conf_threshold)

    # Plot results (YOLO automatically draws bounding boxes and labels)
    res_plotted = results[0].plot()
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    # Display side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    with col2:
        st.subheader("Detected Stones")
        st.image(res_rgb, use_column_width=True)

    # List detections
    st.subheader("Detection Results")
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        st.warning("No kidney stones detected above the confidence threshold. Try lowering the slider.")
    else:
        st.success(f"Found {len(boxes)} potential kidney stone(s):")
        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            st.write(f"**{i+1}. {class_name}** — Confidence: {conf:.2%}")
