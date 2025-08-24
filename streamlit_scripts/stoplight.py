import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import os
import sys
from illuminate import compare_luminosity

# --- Add yolov5 to path ---
YOLOV5_PATH = os.path.join(os.getcwd(), "yolov5")
sys.path.append(YOLOV5_PATH)

# --- Import from local yolov5 ---
from models.yolo import DetectionModel
import torch.serialization
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.augmentations import letterbox
from yolov5.utils.torch_utils import select_device

# --- Configs ---
MODEL_PATH = './weights/stoplight.pt'
IMG_SIZE = 640
CONF_THRESHOLD = 0.4
LUMINOSITY_DIFF_THRESHOLD = 0.01

# --- Load YOLOv5 model with safe unpickling for PyTorch 2.6+ ---
@st.cache_resource
def load_model():
    device = select_device('cpu')
    torch.serialization.add_safe_globals({
        'models.yolo.DetectionModel': DetectionModel
    })
    return DetectMultiBackend(MODEL_PATH, device=device)

model = load_model()

# --- Detection function ---
def detect_brake_lights(img_pil):
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img_input = letterbox(img_bgr, new_shape=IMG_SIZE, stride=32, auto=True)[0]
    img_input = img_input.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img_input = np.ascontiguousarray(img_input)

    img_tensor = torch.from_numpy(img_input).to(model.device).float()
    img_tensor /= 255.0
    img_tensor = img_tensor.unsqueeze(0)

    pred = model(img_tensor, augment=False)
    pred = non_max_suppression(pred, conf_thres=CONF_THRESHOLD)[0]

    boxes = []
    img_disp = img_bgr.copy()

    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], img_bgr.shape).round()
        for *xyxy, conf, cls in pred:
            x1, y1, x2, y2 = map(int, xyxy)
            cropped = img_bgr[y1:y2, x1:x2]
            boxes.append(cropped)
            cv2.rectangle(img_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return boxes, cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)

# --- Utility for status detection ---
def is_functional(status):
    status = status.lower().strip()
    return "functional" in status or "working" in status or "✅" in status

# --- Streamlit UI ---
st.set_page_config(page_title="Brake Light Checker", layout="centered")
st.title("🚗 Brake Light Functionality Detector")

st.markdown("Upload two images of the same car:")
st.markdown("- One with **brake OFF**")
st.markdown("- One with **brake ON**")

col1, col2 = st.columns(2)
with col1:
    off_file = st.file_uploader("Upload Brake OFF Image", type=["jpg", "jpeg", "png"])
with col2:
    on_file = st.file_uploader("Upload Brake ON Image", type=["jpg", "jpeg", "png"])

if off_file and on_file:
    off_img = Image.open(off_file).convert("RGB")
    on_img = Image.open(on_file).convert("RGB")

    st.subheader("🔍 Detection Results")
    off_boxes, off_disp = detect_brake_lights(off_img)
    on_boxes, on_disp = detect_brake_lights(on_img)

    col3, col4 = st.columns(2)
    with col3:
        st.image(off_disp, caption="Brake OFF Detection", use_container_width=True)
    with col4:
        st.image(on_disp, caption="Brake ON Detection", use_container_width=True)

    if len(off_boxes) != len(on_boxes):
        st.error(f"❌ Mismatch in number of detected brake lights. OFF: {len(off_boxes)}, ON: {len(on_boxes)}")
    elif not off_boxes or not on_boxes:
        st.error("❌ Could not detect brake lights in one or both images.")
    else:
        st.subheader("✅ Brake Light Inference Result")
        results = compare_luminosity(off_boxes, on_boxes, threshold=LUMINOSITY_DIFF_THRESHOLD)

        working_count = sum(1 for result in results if is_functional(str(result[4])))

        if working_count == len(results):
            st.success("✅ All brake lights are working.")
        elif working_count == 0:
            st.error("❌ None of the brake lights are working.")
        else:
            st.warning("⚠️ Some brake lights are NOT working.")

        with st.expander("View Detailed Analysis"):
            for idx, off_lum, on_lum, delta, status in results:
                st.markdown(f"**Box {idx}:**")
                st.markdown(f"- 🚫 Brake OFF Brightness: `{off_lum:.2f}`")
                st.markdown(f"- 🟡 Brake ON Brightness: `{on_lum:.2f}`")
                st.markdown(f"- 🔁 Δ: `{delta:.2f}` → **{status}**")
                st.markdown("---")
else:
    st.info("👆 Upload both images to begin.")