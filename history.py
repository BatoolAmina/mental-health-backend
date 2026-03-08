import torch
import pickle
import re
import string
import os
from transformers import RobertaTokenizer, BertTokenizer
from dotenv import load_dotenv
from model_architecture import HybridClassifier

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.getenv("MODEL_PATH", "model/hybrid_model_weights.bin")
ENCODER_PATH = os.getenv("ENCODER_PATH", "model/label_encoder.pkl")

MODEL_PATH = os.path.join(BASE_DIR, MODEL_PATH)
ENCODER_PATH = os.path.join(BASE_DIR, ENCODER_PATH)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

class_names = label_encoder.classes_

model = HybridClassifier(len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
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