from flask import Blueprint, request, jsonify
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from dotenv import load_dotenv
import os
import logging
import jwt
import datetime
import bcrypt
import requests

from utils.security import generate_otp, hash_value, verify_hash, otp_expiry
from utils.email_service import send_otp_email

load_dotenv()

logger = logging.getLogger(__name__)
if not logger.handlers:
    ch = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    level = logging.DEBUG if os.getenv("DEBUG") or os.getenv("FLASK_DEBUG") else logging.INFO
    logger.setLevel(level)

auth_bp = Blueprint("auth", __name__)

SECRET_KEY = os.getenv("SECRET_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

def _make_mongo_client():
    """Create and return a MongoClient, with optional fallback.

    The environment variable ``MONGO_URI`` is used, defaulting to
    ``mongodb://127.0.0.1:27017``.  If the connection fails and the
    URI contains ``+srv``, we attempt a second connection to localhost.
    All events are logged via :data:`logger`.
    """

    uri = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")

    def attempt(u):
        logger.info("attempting MongoDB connection")
        c = MongoClient(u, serverSelectionTimeoutMS=5000)
        c.admin.command("ping")
        logger.info("mongodb ping succeeded")
        return c

    try:
        return attempt(uri)
    except Exception as exc:
        logger.warning("unable to connect to MongoDB: %s", exc)
        if "+srv" in uri:
            fallback = "mongodb://127.0.0.1:27017"
            logger.info("falling back to local MongoDB")
            try:
                return attempt(fallback)
            except Exception as exc2:
                logger.error("fallback also failed: %s", exc2)
        raise

client = _make_mongo_client()

db = client["mental_health_db"]
users = db["users"]
blacklist = db["blacklist"]

users.create_index("email", unique=True)


def create_token(email, provider="local"):
    payload = {
        "email": email,
        "provider": provider,
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def get_ip():
    return request.headers.get("X-Forwarded-For", request.remote_addr)




@auth_bp.route("/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "").strip()

        if not email or not password:
            return jsonify({"error": "Missing credentials"}), 400

        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        otp = generate_otp()
        hashed_otp = hash_value(otp)

        users.insert_one({
            "email": email,
            "password": hashed_pw,
            "provider": "local",
            "email_verified": False,
            "mfa_enabled": True,
            "created_at": datetime.datetime.utcnow(),
            "ip": get_ip(),
            "otp": {
                "code": hashed_otp,
                "expires_at": otp_expiry()
            }
        })

        send_otp_email(email, otp)

        return jsonify({"verification_required": True}), 201

    except DuplicateKeyError:
        return jsonify({"error": "User already exists"}), 409

    except Exception:
        return jsonify({"error": "Signup failed"}), 500




@auth_bp.route("/verify-email", methods=["POST"])
def verify_email():
    try:
        data = request.get_json()
        email = data.get("email", "").strip().lower()
        otp = data.get("otp", "").strip()

        user = users.find_one({"email": email})

        if not user:
            return jsonify({"error": "User not found"}), 404

        if datetime.datetime.utcnow() > user["otp"]["expires_at"]:
            return jsonify({"error": "OTP expired"}), 400

        if not verify_hash(otp, user["otp"]["code"]):
            return jsonify({"error": "Invalid OTP"}), 400

        users.update_one(
            {"email": email},
            {"$set": {"email_verified": True}, "$unset": {"otp": ""}}
        )

        token = create_token(email)
        return jsonify({"token": token}), 200

    except Exception:
        return jsonify({"error": "Verification failed"}), 500




@auth_bp.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "").strip()

        user = users.find_one({"email": email})

        if not user:
            return jsonify({"error": "Invalid credentials"}), 401

        if user.get("provider") != "local":
            return jsonify({"error": "Use Google login"}), 400

        if not bcrypt.checkpw(password.encode(), user["password"].encode()):
            return jsonify({"error": "Invalid credentials"}), 401

        if not user.get("email_verified"):
            return jsonify({"error": "Email not verified"}), 403

        otp = generate_otp()
        hashed_otp = hash_value(otp)

        users.update_one(
            {"email": email},
            {"$set": {
                "otp": {
                    "code": hashed_otp,
                    "expires_at": otp_expiry()
                }
            }}
        )

        send_otp_email(email, otp)

        return jsonify({"mfa_required": True}), 200

    except Exception:
        return jsonify({"error": "Login failed"}), 500




@auth_bp.route("/verify-otp", methods=["POST"])
def verify_otp():
    try:
        data = request.get_json()
        email = data.get("email", "").strip().lower()
        otp = data.get("otp", "").strip()

        user = users.find_one({"email": email})

        if not user:
            return jsonify({"error": "User not found"}), 404

        if datetime.datetime.utcnow() > user["otp"]["expires_at"]:
            return jsonify({"error": "OTP expired"}), 400

        if not verify_hash(otp, user["otp"]["code"]):
            return jsonify({"error": "Invalid OTP"}), 400

        token = create_token(email)
        return jsonify({"token": token}), 200

    except Exception:
        return jsonify({"error": "OTP verification failed"}), 500




from google.oauth2 import id_token
from google.auth.transport import requests as grequests
import uuid

@auth_bp.route("/google-login", methods=["POST"])
def google_login():
    try:
        data = request.get_json()
        google_token = data.get("token")

        if not google_token:
            return jsonify({"error": "Missing Google token"}), 400

        idinfo = id_token.verify_oauth2_token(
            google_token,
            grequests.Request(),
            GOOGLE_CLIENT_ID
        )

        if idinfo["iss"] not in ["accounts.google.com", "https://accounts.google.com"]:
            return jsonify({"error": "Invalid issuer"}), 401

        if not idinfo.get("email_verified"):
            return jsonify({"error": "Google email not verified"}), 403

        email = idinfo["email"].lower()
        sub = idinfo["sub"]

        user = users.find_one({"email": email})

        if user:
            if user.get("provider") != "google":
                return jsonify({"error": "Account exists. Use normal login."}), 400
        else:
            users.insert_one({
                "email": email,
                "provider": "google",
                "google_id": sub,
                "email_verified": True,
                "mfa_enabled": False,
                "created_at": datetime.datetime.utcnow(),
                "ip": get_ip()
            })

        jwt_payload = {
            "email": email,
            "provider": "google",
            "jti": str(uuid.uuid4()),
            "iat": datetime.datetime.utcnow(),
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)
        }

        token = jwt.encode(jwt_payload, SECRET_KEY, algorithm="HS256")

        return jsonify({"token": token}), 200

    except ValueError:
        return jsonify({"error": "Invalid Google token"}), 401

    except Exception:
        return jsonify({"error": "Google authentication failed"}), 500




@auth_bp.route("/logout", methods=["POST"])
def logout():
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Invalid token"}), 401

    token = auth_header.split(" ")[1]

    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        blacklist.insert_one({
            "token": token,
            "expires_at": decoded["exp"]
        })
        return jsonify({"message": "Logged out successfully"}), 200

    except:
        return jsonify({"error": "Invalid token"}), 401

@auth_bp.route('/refresh', methods=['POST'])
def refresh():
    refresh_token = request.json.get('refresh_token')
    if not refresh_token:
        return jsonify({"msg": "Missing refresh token"}), 400
    
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=["HS256"])
        if payload.get('type') != 'refresh':
            return jsonify({"msg": "Invalid token type"}), 401
            
        new_access_token = jwt.encode({
            'email': payload['email'],
            'type': 'access',
            'exp': datetime.utcnow() + timedelta(minutes=15)
        }, SECRET_KEY, algorithm="HS256")
        
        return jsonify({"access_token": new_access_token}), 200
    except:
        return jsonify({"msg": "Refresh token expired"}), 401