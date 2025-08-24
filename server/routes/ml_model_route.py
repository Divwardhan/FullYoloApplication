from flask import Blueprint, request, jsonify
from controllers.ml_model_controller import (
    create_ml_model,
    fetch_all_models,
    fetch_model_by_id,
    update_ml_model,
    delete_ml_model,
)
from middleware.middleware import auth_required
import json
from flask import render_template_string

ml_model_bp = Blueprint('ml_model', __name__)


@ml_model_bp.route('/', methods=['POST'])
def create_model_route():
    """
    Create a new ML model.
    Accepts: multipart/form-data with fields: name, image, type, weight_file (file), description
    """
    print("Received request to create model.")
    name = request.form.get('name')
    image = request.form.get('image')
    model_type = request.form.get('type')
    description = request.form.get('description')
    weight_file = request.files.get('weight_file')  # Optional

    print(name, image , model_type, description, weight_file)

    status, payload = create_ml_model(
        name,
        image,
        model_type,
        weight_file or None,
        description
    )
    return jsonify(payload), status


@ml_model_bp.route('/', methods=['GET'])
# @auth_required  # Uncomment to protect access
def get_models_route():
    """
    Retrieve all ML models.
    """
    status, payload = fetch_all_models()
    return jsonify(payload), status


@ml_model_bp.route('/<int:model_id>', methods=['GET'])
def get_model_route(model_id):
    """
    Retrieve a specific model by ID.
    """
    status, payload = fetch_model_by_id(model_id)
    return jsonify(payload), status


@ml_model_bp.route('/<int:model_id>', methods=['PUT'])
def update_model_route(model_id):
    """
    Update a model by ID.
    Accepts JSON body with any of: name, image, type, weight_file, description
    """
    data = request.get_json() or {}

    status, payload = update_ml_model(
        model_id,
        data.get('name'),
        data.get('image'),
        data.get('type'),
        data.get('weight_file'),
        data.get('description')  # ✅ New field
    )
    return jsonify(payload), status


@ml_model_bp.route('/<int:model_id>', methods=['DELETE'])
def delete_model_route(model_id):
    """
    Delete a model by ID.
    """
    status, payload = delete_ml_model(model_id)
    return jsonify(payload), status



@ml_model_bp.route('/view/<int:model_id>', methods=['GET'])
def view_model_route(model_id):
    """
    Render a simple HTML page showing details of a specific ML model.
    """
    status, model = fetch_model_by_id(model_id)
    
    if status != 200:
        return f"<h2>Error: {model.get('message', 'Model not found')}</h2>", status

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{{ model.name }} - Details</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            img { max-width: 300px; height: auto; }
            .container { border: 1px solid #ccc; padding: 20px; border-radius: 8px; max-width: 600px; margin: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{{ model.name }}</h1>
            <p><strong>Type:</strong> {{ model.type }}</p>
            <p><strong>Description:</strong> {{ model.description or 'No description' }}</p>
            
            {% if model.image %}
                <img src="{{ model.image }}" alt="Model Image">
            {% endif %}
            
            {% if model.weight_file_url %}
                <p><strong>Weight File:</strong> <a href="{{ model.weight_file_url }}" download>Download</a></p>
            {% endif %}
        </div>
    </body>
    </html>
    """
    return render_template_string(html, model=model)

