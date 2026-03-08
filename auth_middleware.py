from flask import request, jsonify
from functools import wraps
import jwt
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")

from auth import _make_mongo_client

client = _make_mongo_client()

db = client["mental_health_db"]
blacklist_collection = db["blacklist"]


def token_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):

        auth_header = request.headers.get("Authorization")

        if not auth_header:
            return jsonify({"error": "Token missing"}), 401

        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Invalid token format"}), 401

        token = auth_header.split(" ")[1]

        try:
            decoded = jwt.decode(
                token,
                SECRET_KEY,
                algorithms=["HS256"]
            )

            if blacklist_collection.find_one({"token": token}):
                return jsonify({"error": "Token revoked"}), 401

            request.user = decoded

        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401

        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

        except Exception:
            return jsonify({"error": "Authentication failed"}), 401

        return f(*args, **kwargs)

    return wrapper