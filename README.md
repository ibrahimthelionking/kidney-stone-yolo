# 🩺 Kidney Stone Detection with YOLOv8

This repository contains an end-to-end object detection pipeline designed to locate kidney stones in CT scan images. 

## The Motivation
Kidney stones are a common and painful condition. Automating their detection in CT scans can assist radiologists in quicker triage and diagnosis. I built this project to explore the intersection of computer vision and medical imaging.

## The Approach
- **Model:** Fine-tuned a pre-trained `YOLOv8n` (Nano) model for 20 epochs using the `ultralytics` library.
- **Dataset:** 1,054 training images and 123 validation images with bounding box annotations.
- **Performance:** Achieved a final validation mAP50 of **0.746 (74.6%)**. Given the small dataset size and the lightweight Nano architecture, this serves as a strong baseline proof-of-concept.
- **App:** Built with **Streamlit**, allowing users to upload a CT scan, adjust the confidence threshold in real-time, and view the annotated bounding boxes.

## Run it Locally
If you want to run this app on your own machine:
1. Clone the repository: `git clone https://github.com/your-username/kidney-stone-yolo.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

*Note: The trained model weights (`best.pt`) are included in this repository.*
