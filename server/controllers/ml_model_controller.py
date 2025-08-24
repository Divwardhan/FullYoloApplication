from models.ml_model import (
    init_model_table,
    create_model,
    get_model_by_id,
    update_model,
    delete_model,
    get_all_models,
)
import requests
import os
from dotenv import load_dotenv
load_dotenv()
from flask import render_template_string
INDIVIDUAL_YOLOS_API = os.getenv("INDIVIDUAL_YOLOS_API")

# Ensure table exists when this module is imported
init_model_table()


def create_ml_model(name: str, image: str, model_type: str, weight_file=None, description: str = None):
    if not name:
        return 400, {"message": "Name required"}
    
    print(f"Creating model with name: {name}, image: {image}, type: {model_type}")
    weight_filename = None

    # Handle uploaded weight file
    if weight_file:
        weight_filename = weight_file.filename
        print(f"Received weight file: {weight_filename}")
        print(f"{weight_file.content_type} -------------- {weight_file.filename} ------------------- {INDIVIDUAL_YOLOS_API}upload_weight/")
        storage_response = requests.post(
            f'{INDIVIDUAL_YOLOS_API}upload_weight',
            files={'file': (weight_filename, weight_file.stream)}
        )

        if storage_response.status_code != 200:
            return 500, {"message": "Weight file upload failed"}

    # Save model metadata including weight file name and description
    model_id = create_model(name, image, model_type, weight_filename, description)

    return 201, {
        "id": model_id,
        "name": name,
        "image": image,
        "type": model_type,
        "weight_file": weight_filename,
        "description": description
    }


def fetch_all_models():
    raw_models = get_all_models()
    models = []
    for row in raw_models:
        model = {
            "id": row[0],
            "name": row[1],
            "image": row[2],
            "weight_file": row[3],
            "type": row[4],
            "description": row[5]  # ✅ Include description
        }
        models.append(model)
    return 200, {"models": models}


def fetch_model_by_id(model_id: int):
    model = get_model_by_id(model_id)
    if not model:
        return 404, {"message": "Model not found"}
    
    id_, name, image,  model_type,weight_file, description = model
    return 200, {
        "id": id_,
        "name": name,
        "image": image,
        "type": model_type,
        "weight_file": weight_file,
        "description": description  # ✅ Include description
    }


def update_ml_model(model_id: int, name: str = None, image: str = None, model_type: str = None, weight_file: str = None, description: str = None):
    updated = update_model(model_id, name, image, model_type, weight_file, description)
    if not updated:
        return 404, {"message": "Model not found"}
    return 200, {"message": "Model updated"}


def delete_ml_model(model_id: int):
    deleted = delete_model(model_id)
    if not deleted:
        return 404, {"message": "Model not found"}
    return 200, {"message": "Model deleted"}
