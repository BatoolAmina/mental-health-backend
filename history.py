import os
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_HOME"] = os.getenv("HF_HOME", "F:/Batool Amina/MHD/huggingface_cache")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.getenv("HF_HOME", "F:/Batool Amina/MHD/huggingface_cache")

import torch
import pickle
import re
import string
from transformers import RobertaTokenizer, BertTokenizer
from huggingface_hub import hf_hub_download, login
from model_architecture import HybridClassifier

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

REPO_ID = os.getenv("HF_REPO_ID", "BatoolAmina/mental-health-chatbot-hybrid")
MODEL_FILE = os.getenv("HF_MODEL_FILE", "hybrid_model_weights.bin")
ENCODER_FILE = os.getenv("HF_ENCODER_FILE", "label_encoder.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
encoder_path = hf_hub_download(repo_id=REPO_ID, filename=ENCODER_FILE)

with open(encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

class_names = label_encoder.classes_

model = HybridClassifier(len(class_names))
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

r_tok = RobertaTokenizer.from_pretrained("roberta-base")
b_tok = BertTokenizer.from_pretrained("bert-base-uncased")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|\@\w+|\#|\d+", "", text)
    return text.translate(str.maketrans("", "", string.punctuation)).strip()

def predict(text):
    text = clean_text(text)

    r = r_tok(
        text,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    b = b_tok(
        text,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    with torch.no_grad():
        output = model(
            r["input_ids"].to(device),
            r["attention_mask"].to(device),
            b["input_ids"].to(device),
            b["attention_mask"].to(device)
        )

        probs = torch.softmax(output, dim=1)
        confidence, index = torch.max(probs, dim=1)

    return {
        "label": class_names[index.item()],
        "confidence": round(confidence.item() * 100, 2)
    }