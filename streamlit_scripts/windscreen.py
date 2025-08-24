import torchvision.transforms as T
import streamlit as st
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import tempfile
import subprocess
import torchvision.models as models
import torch.nn as nn
import datetime 
def get_resnet_model(num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Paths
CNN2_PATH = "./cnn/resnet_crack.pth"
CNN1_PATH = "./cnn/resnet_windscreen_model.pth"
YOLO_PATH = "./yolov5"
YOLO_WEIGHTS = "./weights/WindscreenExp44.pt"

# Load CNN Models
@st.cache_resource
def load_models():
    cnn1 = get_resnet_model(num_classes=2)
    cnn1.load_state_dict(torch.load(CNN1_PATH, map_location=torch.device('cpu')))
    cnn1.eval()

    cnn2 = get_resnet_model(num_classes=2)
    cnn2.load_state_dict(torch.load(CNN2_PATH, map_location=torch.device('cpu')))
    cnn2.eval()

    return cnn1, cnn2

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_cnn(model, img):
    if img.mode != "RGB":
        img = img.convert("RGB")  # ✅ Ensure correct number of channels
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
    return pred


def run_yolov5(image_path):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"result_{timestamp}"
    output_dir = "yolo_output"

    command = [
        "python", f"{YOLO_PATH}/detect.py",
        "--weights", YOLO_WEIGHTS,
        "--img", "640",
        "--conf", "0.25",
        "--source", image_path,
        "--project", output_dir,
        "--name", run_name,
        "--exist-ok"
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL)

    # Return exact output path for this run
    result_path = Path(f"{output_dir}/{run_name}")
    files = sorted(result_path.glob("*.jpg"))
    return files[-1] if files else None



# Streamlit App
st.title("🚘 Windshield Crack Detection App")

st.markdown("""
Upload an image of a vehicle. The app will:
1. Check if there's a windshield
2. Check if the windshield is cracked
3. Detect the crack with YOLOv5 if found
""")

uploaded_files = st.file_uploader(
    "📂 Upload a Folder of Images (multiple selection allowed)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    cnn1, cnn2 = load_models()

    for uploaded_file in uploaded_files:
        st.divider()
        st.subheader(f"📷 Processing: {uploaded_file.name}")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Save temp file
        temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.convert("RGB").save(temp_image.name)

        # Step 1: Windshield detection
        pred1 = predict_cnn(cnn1, image)
        if pred1 == 0:
            st.warning("❌ No windshield detected.")
            continue
        else:
            st.success("✅ Windshield detected.")

        # Step 2: Crack detection
        pred2 = predict_cnn(cnn2, image)
        if pred2 == 1:
            st.info("🪟 Windshield is intact. No cracks detected.")
        else:
            st.success("⚠️ Crack detected on the windshield.")

            # Step 3: Run YOLOv5
            st.write("🔍 Running YOLOv5 to locate the crack...")
            result_img = run_yolov5(temp_image.name)
            if result_img:
                st.image(str(result_img), caption="Crack Localization", use_container_width=True)
            else:
                st.error("No bounding box output found.")