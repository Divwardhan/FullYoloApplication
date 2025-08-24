import streamlit as st
import torch
import numpy as np
import cv2
import os
import sys
import datetime
import subprocess
import tempfile
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn

# --- Add yolov5 to path ---illum
YOLOV5_PATH = os.path.join(os.getcwd(), "yolov5")
sys.path.append(YOLOV5_PATH)

# --- YOLOv5 imports ---
from models.yolo import DetectionModel
import torch.serialization
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.augmentations import letterbox
from yolov5.utils.torch_utils import select_device

# --- Helper function for luminance comparison ---
from illuminate import compare_luminosity

# --- Constants ---
BRAKE_MODEL_PATH = './weights/stoplight.pt'
WINDSCREEN_CNN1_PATH = "./cnn/resnet_windscreen_model.pth"
CRACK_CNN2_PATH = "./cnn/resnet_crack.pth"
YOLO_CRACK_WEIGHTS = "./weights/WindscreenExp44.pt"
IMG_SIZE = 640
CONF_THRESHOLD = 0.4
LUMINOSITY_DIFF_THRESHOLD = 0.01

# --- Load Brake Light YOLOv5 Model ---
@st.cache_resource
def load_brake_model():
    device = select_device('cpu')
    torch.serialization.add_safe_globals({'models.yolo.DetectionModel': DetectionModel})
    return DetectMultiBackend(BRAKE_MODEL_PATH, device=device)

# --- Load CNN models ---
def get_resnet_model(num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

@st.cache_resource
def load_cnn_models():
    cnn1 = get_resnet_model(num_classes=2)
    cnn1.load_state_dict(torch.load(WINDSCREEN_CNN1_PATH, map_location=torch.device('cpu')))
    cnn1.eval()

    cnn2 = get_resnet_model(num_classes=2)
    cnn2.load_state_dict(torch.load(CRACK_CNN2_PATH, map_location=torch.device('cpu')))
    cnn2.eval()

    return cnn1, cnn2

# --- Brake Light Detection ---
def detect_brake_lights(model, img_pil):
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img_input = letterbox(img_bgr, new_shape=IMG_SIZE, stride=32, auto=True)[0]
    img_input = img_input.transpose((2, 0, 1))[::-1]
    img_input = np.ascontiguousarray(img_input)

    img_tensor = torch.from_numpy(img_input).to(model.device).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    pred = model(img_tensor, augment=False)
    pred = non_max_suppression(pred, conf_thres=CONF_THRESHOLD)[0]

    boxes = []
    img_disp = img_bgr.copy()
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], img_bgr.shape).round()
        for *xyxy, _, _ in pred:
            x1, y1, x2, y2 = map(int, xyxy)
            boxes.append(img_bgr[y1:y2, x1:x2])
            cv2.rectangle(img_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return boxes, cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)

# --- Windshield Detection Logic ---
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_cnn(model, img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
    return pred

def run_yolov5_detection(image_path):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"result_{timestamp}"
    output_dir = "yolo_output"

    command = [
        "python", f"{YOLOV5_PATH}/detect.py",
        "--weights", YOLO_CRACK_WEIGHTS,
        "--img", "640",
        "--conf", "0.25",
        "--source", image_path,
        "--project", output_dir,
        "--name", run_name,
        "--exist-ok"
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL)

    result_path = Path(f"{output_dir}/{run_name}")
    files = sorted(result_path.glob("*.jpg"))
    return files[-1] if files else None

# --- UI Layout ---
st.set_page_config(page_title="Hybrid Vehicle Analysis", layout="wide")
st.title("🧠 Hybrid Vehicle Analysis Platform")

app_mode = st.sidebar.selectbox("Choose Model", ["Brake Light Detector", "Windshield Crack Detector"])

if app_mode == "Brake Light Detector":
    st.header("🚗 Brake Light Functionality Detector")
    model = load_brake_model()

    col1, col2 = st.columns(2)
    with col1:
        off_file = st.file_uploader("Upload Brake OFF Image", type=["jpg", "jpeg", "png"], key="off")
    with col2:
        on_file = st.file_uploader("Upload Brake ON Image", type=["jpg", "jpeg", "png"], key="on")

    if off_file and on_file:
        off_img = Image.open(off_file).convert("RGB")
        on_img = Image.open(on_file).convert("RGB")

        off_boxes, off_disp = detect_brake_lights(model, off_img)
        on_boxes, on_disp = detect_brake_lights(model, on_img)

        col3, col4 = st.columns(2)
        with col3:
            st.image(off_disp, caption="Brake OFF Detection", use_container_width=True)
        with col4:
            st.image(on_disp, caption="Brake ON Detection", use_container_width=True)

        if not off_boxes or not on_boxes:
            st.error("❌ Could not detect brake lights in one or both images.")
        else:
            st.subheader("📊 Brightness Comparison")
            results = compare_luminosity(off_boxes, on_boxes, threshold=LUMINOSITY_DIFF_THRESHOLD)
            for idx, off_lum, on_lum, delta, status in results:
                st.markdown(f"**Box {idx}:**")
                st.markdown(f"- 🚫 Brake OFF Brightness: `{off_lum:.2f}`")
                st.markdown(f"- 🟡 Brake ON Brightness: `{on_lum:.2f}`")
                st.markdown(f"- 🔁 Δ: `{delta:.2f}` → **{status}**")
                st.markdown("---")

elif app_mode == "Windshield Crack Detector":
    st.header("🚘 Windshield Crack Detection")
    uploaded_files = st.file_uploader(
        "📂 Upload Images of Vehicles",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        cnn1, cnn2 = load_cnn_models()

        for uploaded_file in uploaded_files:
            st.divider()
            st.subheader(f"📷 Processing: {uploaded_file.name}")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            image.convert("RGB").save(temp_image.name)

            pred1 = predict_cnn(cnn1, image)
            if pred1 == 0:
                st.warning("❌ No windshield detected.")
                continue
            else:
                st.success("✅ Windshield detected.")

            pred2 = predict_cnn(cnn2, image)
            if pred2 == 1:
                st.info("🪟 Windshield is intact. No cracks detected.")
            else:
                st.success("⚠️ Crack detected on the windshield.")
                st.write("🔍 Running YOLOv5 to locate the crack...")
                result_img = run_yolov5_detection(temp_image.name)
                if result_img:
                    st.image(str(result_img), caption="Crack Localization", use_container_width=True)
                else:
                    st.error("No bounding box output found.")
