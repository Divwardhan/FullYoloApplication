from functools import wraps
from flask import request, jsonify
from flask_jwt_extended import decode_token
from models.user_model import get_user_by_email


def auth_required(fn):
    """Authenticate via token header and attach email to request."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        token = request.headers.get("token")
        # print(token)
        if not token:
            return jsonify({"message": "Missing token"}), 401
        try:
            decoded = decode_token(token)
            email = decoded.get("sub")
            # print(email)
        except Exception:
            return jsonify({"message": "Invalid token"}), 401

        if not email or not get_user_by_email(email):
            return jsonify({"message": "Unauthorized"}), 401

        request.email = email
        return fn(*args, **kwargs)

    return wrapper
