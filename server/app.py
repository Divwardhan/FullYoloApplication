from flask import Flask,jsonify
import os
from dotenv import load_dotenv
from flask_jwt_extended import JWTManager
from models.db import test_connection
from models.user_model import init_db
from routes.user_route import user_bp
from routes.ml_model_route import ml_model_bp
from routes.inference_route import inference_bp
from flask_cors import CORS

load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "super-secret")
app.config["JWT_HEADER_NAME"] = "token"
app.config["JWT_HEADER_TYPE"] = ""  # No prefix like Bearer
jwt = JWTManager(app)

init_db()
app.register_blueprint(user_bp , url_prefix='/api/user')
app.register_blueprint(ml_model_bp, url_prefix='/api/models')
app.register_blueprint(inference_bp, url_prefix='/api/inference')


# Upload settings
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return 'YOLOv5 Flask Backend is Running'


@app.route("/cors-test")
def test_cors():
    return jsonify({"ok": True})

@app.route("/api/user/test-cors", methods=["POST"])
def test_cors_again():
    return jsonify({"message": "POST works and CORS is fine"})


@app.route('/test-db')
def test_db():
    """Check database connectivity by running a simple query."""
    if test_connection():
        return jsonify({'connected': True})
    return jsonify({'connected': False}), 500


if __name__ == '__main__':
    app.run(debug=True)
