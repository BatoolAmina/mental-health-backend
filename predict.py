import os
from dotenv import load_dotenv

# MUST happen before importing transformers or huggingface_hub
load_dotenv()
os.environ["HF_HOME"] = os.getenv("HF_HOME", "F:/Batool Amina/MHD/huggingface_cache")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.getenv("HF_HOME", "F:/Batool Amina/MHD/huggingface_cache")

import torch
import pickle
import re
import string
from transformers import RobertaTokenizer, BertTokenizer
from huggingface_hub import hf_hub_download, login
from model_architecture import UltraHybridClassifier
from auth import _make_mongo_client
from datetime import datetime
from logic_rules import apply_logic_rules

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

REPO_ID = os.getenv("HF_REPO_ID", "BatoolAmina/mental-health-chatbot-hybrid")
MODEL_FILE = os.getenv("HF_MODEL_FILE", "hybrid_model_weights.bin")
ENCODER_FILE = os.getenv("HF_ENCODER_FILE", "label_encoder.pkl")

# Use cuda if available, otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
encoder_path = hf_hub_download(repo_id=REPO_ID, filename=ENCODER_FILE)

with open(encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

class_names = label_encoder.classes_
model = UltraHybridClassifier(len(class_names))
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

r_tok = RobertaTokenizer.from_pretrained("roberta-base")
b_tok = BertTokenizer.from_pretrained("bert-base-uncased")

client = _make_mongo_client()
db = client["mental_health_db"]
history_collection = db["chat_history"]
corrections_collection = db["model_corrections"]
memory_collection = db["prediction_memory"]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+|\@\w+|\#|\d+', '', text)
    text = re.sub(r"\s+", " ", text)
    return text.translate(str.maketrans("", "", string.punctuation)).strip()

def check_learned_memory(text, email):
    if email:
        match = corrections_collection.find_one({"email": email, "text": text.lower()})
        if match:
            return {
                "label": match["correct_label"],
                "confidence": 100.0,
                "risk_level": "low" if match["correct_label"] in ["normal", "happy"] else "medium",
                "source": "private_learned_memory"
            }
    
    global_match = memory_collection.find_one({"text": text.lower(), "learned_globally": True})
    if global_match:
        return {
            "label": global_match["label"],
            "confidence": 100.0,
            "risk_level": global_match["risk_level"],
            "source": "global_consensus_memory"
        }
    return None

def get_user_context(email):
    recent_chats = history_collection.find({"email": email}).sort("updated_at", -1).limit(1)
    context_str = ""
    for chat in recent_chats:
        for msg in chat.get("messages", [])[-5:]:
            if msg["sender"] == "user":
                context_str += msg["text"] + " "
    return context_str.strip()

def predict(text, email=None):
    try:
        cleaned = clean_text(text)
        
        memory_result = check_learned_memory(cleaned, email)
        if memory_result:
            return memory_result

        if len(cleaned.split()) < 3 and ("i am" in cleaned or "my name" in cleaned):
             return {
                "label": "normal",
                "confidence": 100.0,
                "risk_level": "low",
                "source": "identity_logic"
            }

        past_context = ""
        if email:
            past_context = get_user_context(email)
        
        input_text = f"{past_context} {cleaned}".strip() if past_context else cleaned

        r = r_tok(input_text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
        b = b_tok(input_text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")

        with torch.no_grad():
            output = model(
                r["input_ids"].to(device), 
                r["attention_mask"].to(device), 
                b["input_ids"].to(device), 
                b["attention_mask"].to(device)
            )
            probs = torch.softmax(output, dim=1)
            confidence, index = torch.max(probs, dim=1)

        raw_label = class_names[index.item()]
        raw_score = round(confidence.item() * 100, 2)

        final_label, final_score, forced_risk = apply_logic_rules(cleaned, raw_label, raw_score)

        risk = forced_risk if forced_risk else "low"
        if not forced_risk:
            if final_label == "suicidal": risk = "high"
            elif final_label in ["depression", "anxiety"]: risk = "medium"

        memory_collection.update_one(
            {"text": cleaned},
            {
                "$set": {
                    "label": final_label,
                    "risk_level": risk,
                    "timestamp": datetime.utcnow()
                }
            },
            upsert=True
        )

        return {
            "label": final_label,
            "confidence": final_score,
            "risk_level": risk,
            "source": "hybrid_model_with_rules"
        }

    except Exception:
        return {"label": "normal", "confidence": 0.0, "risk_level": "low"}