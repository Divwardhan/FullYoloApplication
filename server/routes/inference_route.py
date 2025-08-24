from flask import Blueprint, request, jsonify
from controllers.inference_controller import run_inference , run_inference_windscreen
from middleware.middleware import auth_required
from flask import render_template_string
from models.inference_result_model import get_all_inference_results

inference_bp = Blueprint('inference', __name__)


@inference_bp.route('/<int:model_id>', methods=['POST'])
@auth_required
def inference_route(model_id):
    if 'image' not in request.files:
        return jsonify({'message': 'No image provided'}), 400
    image = request.files['image']
    print("Heelo")
    status, payload = run_inference(model_id, image, request.email)
    return jsonify(payload), status

@inference_bp.route('/windscreen', methods=['POST'])
@auth_required
def run_windscreen_inference():
    """
    Endpoint to run inference on a windscreen image.
    Expects 'image' file in the request.
    """
    if 'image' not in request.files:
        return jsonify({'message': 'No image provided'}), 400
    
    image = request.files['image']
    

    status, payload = run_inference_windscreen('999', image , request.email)
    return jsonify(payload), status

@inference_bp.route('/stoplight' , methods=['POST'])
@auth_required
def run_stoplight_inference():
    """
    Endpoint to run inference on a stoplight image.
    Expects 'image' file in the request.
    """
    if 'image1' and "image2" not in request.files:
        return jsonify({'message': 'No image provided'}), 400
    
    image1 = request.files['image1']
    image2 = request.files['image2']

    
    status, payload = run_inference_stoplight('999', image1 , image2 , request.email)
    return jsonify(payload), status

@inference_bp.route('/view', methods=['GET'])
def view_all_inference_results():
    results = get_all_inference_results()

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Inference Results</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:hover { background-color: #f9f9f9; }
            a { color: blue; text-decoration: underline; }
        </style>
    </head>
    <body>
        <h2>All Inference Results</h2>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Model</th>
                    <th>User</th>
                    <th>Input Image</th>
                    <th>Output Image</th>
                    <th>JSON Response</th>
                </tr>
            </thead>
            <tbody>
                {% for result in results %}
                <tr>
                    <td>{{ result.id }}</td>
                    <td>{{ result.model_name }} (ID: {{ result.model_id }})</td>
                    <td>{{ result.user_email }}</td>
                    <td><a href="{{ result.input_image_url }}" target="_blank">Input</a></td>
                    <td><a href="{{ result.output_image_url }}" target="_blank">Output</a></td>
                    <td><pre style="white-space: pre-wrap;">{{ result.json_response }}</pre></td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </body>
    </html>
    """

    return render_template_string(html_template, results=results)
