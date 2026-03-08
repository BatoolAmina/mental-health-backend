from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
from bson import ObjectId
import os
from datetime import datetime
import json

from predict import predict
from auth import auth_bp
from auth_middleware import token_required

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

app.register_blueprint(auth_bp, url_prefix="/auth")

from auth import _make_mongo_client

client = _make_mongo_client()
db = client["mental_health_db"]
history_collection = db["chat_history"]
corrections_collection = db["model_corrections"]
memory_collection = db["prediction_memory"]

def get_current_user():
    return request.user["email"]

def serialize_chat(chat):
    if "_id" in chat:
        chat["_id"] = str(chat["_id"]) if not isinstance(chat["_id"], str) else chat["_id"]
    return chat

def find_chat_by_id(chat_id, email):
    chat = history_collection.find_one({
        "_id": chat_id,
        "email": email
    })
    if chat:
        return chat
    
    try:
        chat = history_collection.find_one({
            "_id": ObjectId(chat_id),
            "email": email
        })
        return chat
    except:
        return None

@app.route("/predict", methods=["POST"])
@token_required
def predict_route():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        conversation_id = data.get("conversation_id")

        if not text:
            return jsonify({"error": "Text field is required"}), 400

        email = get_current_user()
        
        result = predict(text, email=email)

        message_pair = [
            {
                "sender": "user",
                "text": text,
                "timestamp": datetime.utcnow()
            },
            {
                "sender": "bot",
                "text": f"{result['label']} ({result['confidence']}%)",
                "label": result["label"],
                "confidence": result["confidence"],
                "risk_level": result.get("risk_level", "low"),
                "timestamp": datetime.utcnow()
            }
        ]

        if conversation_id:
            history_collection.update_one(
                {"_id": ObjectId(conversation_id), "email": email},
                {
                    "$push": {"messages": {"$each": message_pair}},
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
        else:
            new_chat = history_collection.insert_one({
                "email": email,
                "title": text[:40],
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "messages": message_pair
            })
            conversation_id = str(new_chat.inserted_id)

        return jsonify({
            "success": True,
            "label": result["label"],
            "confidence": result["confidence"],
            "risk_level": result.get("risk_level", "low"),
            "conversation_id": conversation_id
        }), 200

    except Exception:
        return jsonify({"error": "Prediction failed"}), 500

@app.route("/feedback", methods=["POST"])
@token_required
def submit_feedback():
    try:
        data = request.get_json()
        text = data.get("text", "").strip().lower()
        incorrect_label = data.get("incorrect_label")
        correct_label = data.get("correct_label")
        
        if not text or not correct_label:
            return jsonify({"error": "Text and correct label are required"}), 400

        email = get_current_user()

        corrections_collection.update_one(
            {"email": email, "text": text},
            {
                "$set": {
                    "incorrect_label": incorrect_label,
                    "correct_label": correct_label,
                    "timestamp": datetime.utcnow()
                }
            },
            upsert=True
        )

        global_agreement_count = corrections_collection.count_documents({
            "text": text,
            "correct_label": correct_label
        })

        is_globally_learned = False
        if global_agreement_count >= 3:
            risk = "low"
            if correct_label == "suicidal": risk = "high"
            elif correct_label in ["depression", "anxiety"]: risk = "medium"

            memory_collection.update_one(
                {"text": text},
                {
                    "$set": {
                        "label": correct_label,
                        "risk_level": risk,
                        "learned_globally": True,
                        "confidence_score": 100.0,
                        "last_updated": datetime.utcnow()
                    }
                },
                upsert=True
            )
            is_globally_learned = True

        return jsonify({
            "success": True,
            "message": "Feedback recorded",
            "globally_learned": is_globally_learned
        }), 200

    except Exception:
        return jsonify({"error": "Feedback submission failed"}), 500

@app.route("/predict-stream", methods=["POST"])
@token_required
def predict_stream():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        conversation_id = data.get("conversation_id")

        if not text:
            return jsonify({"error": "Text required"}), 400

        email = get_current_user()
        result = predict(text, email=email)
        response_text = f"{result['label']} ({result['confidence']}%)"

        def generate():
            for char in response_text:
                yield char

            message_pair = [
                {
                    "sender": "user",
                    "text": text,
                    "timestamp": datetime.utcnow()
                },
                {
                    "sender": "bot",
                    "text": response_text,
                    "label": result["label"],
                    "confidence": result["confidence"],
                    "timestamp": datetime.utcnow()
                }
            ]
        
            if conversation_id:
                history_collection.update_one(
                    {"_id": ObjectId(conversation_id), "email": email},
                    {
                        "$push": {"messages": {"$each": message_pair}},
                        "$set": {"updated_at": datetime.utcnow()}
                    }
                )
            else:
                new_chat = history_collection.insert_one({
                    "email": email,
                    "title": text[:40],
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    "messages": message_pair
                })

        return Response(generate(), mimetype="text/plain")

    except Exception:
        return jsonify({"error": "Streaming failed"}), 500

@app.route("/history", methods=["GET"])
@token_required
def history_route():
    try:
        email = get_current_user()
        all_flag = request.args.get("all", "false").lower() == "true"
        if all_flag:
            chats = list(
                history_collection.find({"email": email}, {"messages": 0}).sort("updated_at", -1)
            )
            chats = [serialize_chat(chat) for chat in chats]
            return jsonify({
                "chats": chats,
                "has_more": False
            }), 200

        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 10))
        skip = (page - 1) * limit

        chats = list(
            history_collection.find({"email": email}, {"messages": 0}).sort("updated_at", -1).skip(skip).limit(limit)
        )

        total = history_collection.count_documents({"email": email})
        has_more = skip + limit < total

        chats = [serialize_chat(chat) for chat in chats]

        return jsonify({
            "chats": chats,
            "has_more": has_more
        }), 200

    except Exception:
        return jsonify({"error": "Unable to fetch history"}), 500

@app.route("/history/<chat_id>", methods=["GET"])
@token_required
def single_chat(chat_id):
    try:
        email = get_current_user()
        chat = find_chat_by_id(chat_id, email)

        if not chat:
            return jsonify({"error": "Chat not found"}), 404

        return jsonify(serialize_chat(chat)), 200

    except Exception as e:
        print(f"Error loading chat: {str(e)}")
        return jsonify({"error": "Unable to load chat"}), 500

@app.route("/history/<chat_id>", methods=["DELETE"])
@token_required
def delete_chat(chat_id):
    try:
        email = get_current_user()
        chat = find_chat_by_id(chat_id, email)

        if not chat:
            return jsonify({"error": "Chat not found"}), 404

        history_collection.delete_one({"_id": chat["_id"]})

        return jsonify({"success": True}), 200

    except Exception:
        return jsonify({"error": "Delete failed"}), 500

@app.route("/history/<chat_id>", methods=["PATCH"])
@token_required
def rename_chat(chat_id):
    try:
        email = get_current_user()
        title = request.json.get("title", "").strip()
        chat = find_chat_by_id(chat_id, email)

        if not chat:
            return jsonify({"error": "Chat not found"}), 404

        history_collection.update_one(
            {"_id": chat["_id"]},
            {"$set": {"title": title}}
        )

        return jsonify({"success": True}), 200

    except Exception:
        return jsonify({"error": "Rename failed"}), 500

@app.route("/history/<chat_id>/edit-message", methods=["POST"])
@token_required
def edit_message(chat_id):
    try:
        email = get_current_user()
        data = request.get_json()
        message_index = data.get("message_index")
        edited_text = data.get("text", "").strip()

        if not edited_text or message_index is None:
            return jsonify({"error": "Missing data"}), 400

        chat = find_chat_by_id(chat_id, email)
        if not chat:
            return jsonify({"error": "Chat not found"}), 404

        messages = chat.get("messages", [])
        new_messages = messages[:message_index]
        
        new_messages.append({
            "sender": "user",
            "text": edited_text,
            "timestamp": datetime.utcnow()
        })

        result = predict(edited_text, email=email)
        
        bot_message = {
            "sender": "bot",
            "text": f"{result['label']} ({result['confidence']}%)",
            "label": result["label"],
            "confidence": result["confidence"],
            "risk_level": result.get("risk_level", "low"),
            "timestamp": datetime.utcnow()
        }
        new_messages.append(bot_message)

        history_collection.update_one(
            {"_id": chat["_id"]},
            {"$set": {"messages": new_messages, "updated_at": datetime.utcnow()}}
        )

        return jsonify({
            "success": True,
            "messages": new_messages,
            "label": result["label"]
        }), 200

    except Exception:
        return jsonify({"error": "Edit failed"}), 500

@app.route("/history/<chat_id>/update-messages", methods=["POST"])
@token_required
def update_messages(chat_id):
    try:
        email = get_current_user()
        data = request.get_json()
        new_messages = data.get("messages")

        if new_messages is None:
            return jsonify({"error": "Messages list required"}), 400

        chat = find_chat_by_id(chat_id, email)
        if not chat:
            return jsonify({"error": "Chat not found"}), 404

        history_collection.update_one(
            {"_id": chat["_id"]},
            {
                "$set": {
                    "messages": new_messages,
                    "updated_at": datetime.utcnow()
                }
            }
        )

        return jsonify({"success": True}), 200

    except Exception:
        return jsonify({"error": "Update failed"}), 500

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "Backend running",
        "service": "Mental Health Detection API"
    }), 200

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=True
    )