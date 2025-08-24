import torchvision.transforms as T
import torch
import sys
import os
import json
from pathlib import Path
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, send_file
import traceback
import tempfile
import torchvision.models as models
import torch.nn as nn
import cv2 

# Initialize Flask app
app = Flask(__name__)

# Add yolov5 path
YOLOV5_PATH = os.path.join(os.path.dirname(__file__), 'yolov5')
sys.path.append(YOLOV5_PATH)

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.augmentations import letterbox



def get_resnet_model(num_classes):
    model = models.resnet18(weights=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
CNN2_PATH = "./cnn/resnet_crack.pth"
CNN1_PATH = "./cnn/resnet_windscreen_model.pth"
STOPLIGHT_MODEL_PATH = './pt_files/stoplight.pt'
IMG_SIZE = 640
CONF_THRESHOLD = 0.4
LUMINOSITY_DIFF_THRESHOLD = 0.01

# Load CNN Models
def load_models():
    cnn1 = get_resnet_model(num_classes=2)
    cnn1.load_state_dict(torch.load(CNN1_PATH, map_location=torch.device('cpu')))
    cnn1.eval()

    cnn2 = get_resnet_model(num_classes=2)
    cnn2.load_state_dict(torch.load(CNN2_PATH, map_location=torch.device('cpu')))
    cnn2.eval()

    return cnn1, cnn2


# Load YOLOv5 model
def load_model():
    device = select_device('cpu')
    return DetectMultiBackend(STOPLIGHT_MODEL_PATH, device=device)

model = load_model()

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


def scale_coords(img_shape_resized, coords, img_shape_original):
    gain_w = img_shape_original[1] / img_shape_resized[1]
    gain_h = img_shape_original[0] / img_shape_resized[0]
    coords[:, [0, 2]] *= gain_w
    coords[:, [1, 3]] *= gain_h
    return coords

def run_yolo_inference(image_path, model_path):
    device = select_device('')
    model = DetectMultiBackend(model_path, device=device, dnn=False)
    stride, names = model.stride, model.names
    imgsz = (640, 640)

    # Load original image
    original_img = Image.open(image_path).convert("RGB")
    original_array = np.array(original_img)
    original_shape = original_array.shape  # HWC

    # Resize for model
    resized_img = original_img.resize(imgsz)
    resized_array = np.array(resized_img)
    resized_shape = resized_array.shape

    img = torch.from_numpy(resized_array).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0).to(device)

    # Inference
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    labels = []
    annotator = Annotator(original_array.copy(), line_width=2)

    for i, det in enumerate(pred):
        if det is not None and len(det):
            # Scale to original shape
            det[:, :4] = scale_coords(imgsz, det[:, :4], original_shape)

            for *xyxy, conf, cls in reversed(det):
                label_text = f'{names[int(cls)]} {conf:.2f}'
                xyxy_int = [int(x.item()) for x in xyxy]
                annotator.box_label(xyxy_int, label_text, color=colors(int(cls), True))
                labels.append({
                    "class": names[int(cls)],
                    "confidence": float(conf),
                    "box": xyxy_int
                })

    annotated_img = annotator.result()
    annotated_pil = Image.fromarray(annotated_img)

    return annotated_pil, labels

# Calculate luminosity
def calculate_luminosity(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray_img)

# Detect brake lights and crop boxes
def detect_brake_lights(img_pil):
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img_input = letterbox(img_bgr, new_shape=IMG_SIZE, stride=32, auto=True)[0]
    img_input = img_input.transpose((2, 0, 1))[::-1]
    img_input = np.ascontiguousarray(img_input)

    img_tensor = torch.from_numpy(img_input).to(model.device).float()
    img_tensor /= 255.0
    img_tensor = img_tensor.unsqueeze(0)

    pred = model(img_tensor, augment=False)
    pred = non_max_suppression(pred, conf_thres=CONF_THRESHOLD)[0]

    boxes = []
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], img_bgr.shape).round()
        for *xyxy, conf, cls in pred:
            x1, y1, x2, y2 = map(int, xyxy)
            cropped = img_bgr[y1:y2, x1:x2]
            boxes.append(cropped)

    return boxes

# Compare luminosity
def compare_luminosity(off_boxes, on_boxes, threshold=LUMINOSITY_DIFF_THRESHOLD):
    results = []
    for idx, (off_img, on_img) in enumerate(zip(off_boxes, on_boxes), 1):
        off_lum = calculate_luminosity(off_img)
        on_lum = calculate_luminosity(on_img)
        delta = abs(on_lum - off_lum)
        status = "Functional" if delta >= threshold else "Non-functional"
        results.append({
            "box": idx,
            "off_lum": off_lum,
            "on_lum": on_lum,
            "delta": delta,
            "status": status
        })
    return results

# Main function to integrate with Flask
def check_brake_lights(off_image_path, on_image_path):
    off_img = Image.open(off_image_path).convert("RGB")
    on_img = Image.open(on_image_path).convert("RGB")

    off_boxes = detect_brake_lights(off_img)
    on_boxes = detect_brake_lights(on_img)

    if not off_boxes or not on_boxes:
        return {"error": "Could not detect brake lights in one or both images."}

    results = compare_luminosity(off_boxes, on_boxes)
    return results







@app.route('/infer_stoplight', methods=['POST'])
def infer_stoplight():
    if 'file1' and "file2" not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    image_file1 = request.files['file1']
    image_file2 = request.files['file2']
    if image_file1.filename == '':
        return jsonify({'error': 'No selected image.'}), 400
    try:
        # Save the uploaded image temporarily
        temp_image1 = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image_file1.save(temp_image1.name)
        
        temp_image2 = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image_file2.save(temp_image2.name)

        # Run inference
        results = check_brake_lights(temp_image1.name, temp_image2.name)  # Using the same image for both off and on states
        
        # Clean up temporary file
        os.remove(temp_image1.name)
        os.remove(temp_image2.name)
        
        return jsonify(results), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/infer_crack', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['file']
    image = Image.open(image_file.stream)

    temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.convert("RGB").save(temp_image.name)
    

    response = {
        'windscreen_detected': False,
        'crack_detected': False,
        'labels': [],
        'image_url': None
    }

    # Step 1: Windshield detection
    pred1 = predict_cnn(cnn1, image)
    if pred1 == 0:
        return jsonify(response)

    response['windscreen_detected'] = True

    # Step 2: Crack detection
    pred2 = predict_cnn(cnn2, image)
    if pred2 == 1:
        return jsonify(response)

    response['crack_detected'] = True

    # Step 3: YOLO Detection
    

    try:
        result_img, labels = run_yolo_inference(temp_image.name, 'pt_files/WindscreenExp44.pt')
        response['labels'] = labels
    except Exception as e:
        print(f"[ERROR] Inference failed: {str(e)}")
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    # Step 5: Save annotated image
    output_path = os.path.join('..','uploads', f'annotated_{image_file.filename}')
    print(f"[INFO] Saving annotated image to: {output_path}")
    try:
        result_img.save(output_path)
        print(f"[INFO] Annotated image saved to: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save annotated image: {str(e)}")
        return jsonify({"error": "Failed to save annotated image."}), 500

    response['image_url'] = f"/uploads/annotated_{image_file.filename}"
    print(response)

    return jsonify(response)

@app.route('/infer', methods=['POST'])
def infer_image():
    """
    Endpoint to process an image and run YOLO inference.
    Accepts image (as file) and weight filename (as text).
    """
    print("[INFO] Received /infer request")

    # Step 1: Validate request fields
    if 'file' not in request.files:
        print("[ERROR] 'file' not in request")
        return jsonify({"error": "Image file is required."}), 400

    if 'weights' not in request.form:
        print("[ERROR] 'weights' not in request")
        return jsonify({"error": "Weight filename is required."}), 400

    image_file = request.files['file']
    weight_filename = request.form['weights'].strip()

    print(weight_filename)

    if image_file.filename == '':
        print("[ERROR] Image filename is empty")
        return jsonify({"error": "No selected image."}), 400

    if weight_filename == '':
        print("[ERROR] Weight filename is empty")
        return jsonify({"error": "No weight filename provided."}), 400

    try:
        # Step 2: Prepare directories and paths
        os.makedirs('uploads', exist_ok=True)
        image_path = os.path.join('uploads', image_file.filename)
        image_file.save(image_path)
        print(f"[INFO] Saved image to: {image_path}")

        # Step 3: Resolve weight file path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(script_dir, 'pt_files')
        weight_path = os.path.join(weights_dir, weight_filename)

        if not os.path.isfile(weight_path):
            print(f"[ERROR] Weights file not found: {weight_path}")
            return jsonify({"error": f"Weights file not found: {weight_path}"}), 404

        print(f"[INFO] Using weights from: {weight_path}")

        # Step 4: Run inference
        try:
            annotated_img, labels_json = run_yolo_inference(image_path, weight_path)
        except Exception as e:
            print(f"[ERROR] Inference failed: {str(e)}")
            return jsonify({"error": f"Inference failed: {str(e)}"}), 500

        # Step 5: Save annotated image
        output_path = os.path.join('..','uploads', f'annotated_{image_file.filename}')
        print(f"[INFO] Saving annotated image to: {output_path}")
        try:
            annotated_img.save(output_path)
            print(f"[INFO] Annotated image saved to: {output_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save annotated image: {str(e)}")
            return jsonify({"error": "Failed to save annotated image."}), 500

        # Step 6: Respond
        return jsonify({
            "labels": labels_json,
            "image_url": f"/uploads/annotated_{image_file.filename}"
        }), 200

    except Exception as e:
        print(f"[ERROR] Unhandled error: {str(e)}")
        return jsonify({
            "error": "Unexpected error occurred.",
            "details": str(e)
        }), 500




@app.route('/uploads/<filename>', methods=['GET'])
def get_image(filename):
    """
    Endpoint to serve the annotated image.
    """
    try:
        return send_file(os.path.join('../uploads', filename), mimetype='image/jpeg')
    except FileNotFoundError:
        return jsonify({"error": "Image not found"}), 404
    

@app.route("/upload_weight" , methods=["POST"])
def upload_weight():
    """
    Endpoint to upload a weight file.
    """
    print("Received weight upload request")
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    print(request.files)
    weight_file = request.files['file']
    
    if weight_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not weight_file.filename.endswith('.pt'):
            return jsonify({"error": "Only .pt files are allowed"}), 400

    try:
        # Create uploads directory if it doesn't exist
        os.makedirs('pt_files', exist_ok=True)

        # Save the uploaded weight file
        weight_path = os.path.join('pt_files', weight_file.filename)
        weight_file.save(weight_path)

        return jsonify({"message": "Weight file uploaded successfully", "filename": weight_file.filename}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test_numpy')
def test_numpy():
    try:
        import numpy
        return f"Numpy version: {numpy.__version__}", 200
    except Exception as e:
        return f"Failed: {str(e)}", 500




if __name__ == '__main__':
    # Ensure the uploads directory exists
    os.makedirs('../uploads', exist_ok=True)
    # Load models
    cnn1, cnn2 = load_models()
    print("Models loaded successfully")
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5001)