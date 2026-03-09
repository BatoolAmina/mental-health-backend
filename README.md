# MHC AI: Mental Health Backend

This repository contains the **Flask-based neural processing engine** for the MHC application. The service manages high-precision emotional classification, secure JWT authentication, and persistent chat history within a MongoDB environment.

---

## Neural Architecture

The system implements a sophisticated **Ultra-Hybrid Classifier** that ensembles multiple deep learning architectures to ensure high accuracy in mental health state detection:

* **BERT**: Extracts deep semantic relationships and linguistic patterns from user input.
* **RoBERTa**: Enhances the semantic core by detecting subtle emotional nuances and sentiment intensity.
* **Bi-LSTM**: A 3-layer sequential processor that analyzes the temporal flow and contextual progression of thoughts.
* **Multi-Head Contextual Attention**: A custom layer that weights specific emotional triggers within the combined transformer hidden states.

---

## Environment Variables

The backend utilizes `python-dotenv` to manage sensitive configuration. Create a `.env` file in the root directory with the following parameters:

### Core System
- `SECRET_KEY`: Cryptographic secret used to sign and verify JWT tokens.
- `MONGO_URI`: Connection string for MongoDB Atlas or a local instance.
- `DEBUG`: Set to `True` for verbose logging and active development diagnostics.

### Neural Vault (Hugging Face)
- `HF_TOKEN`: A Write access token from Hugging Face to authorize model downloads.
- `HF_REPO_ID`: The repository path (e.g., `BatoolAmina/mental-health-chatbot-hybrid`).
- `HF_MODEL_FILE`: The filename of your weights (e.g., `hybrid_model_weights.bin`).
- `HF_ENCODER_FILE`: The filename of your serialized label encoder (e.g., `label_encoder.pkl`).

---

## 🚀 Running the Server

### 1. Environment Setup

Ensure your virtual environment is active and all dependencies, including `huggingface_hub` and `transformers`, are installed.

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Activate virtual environment (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Neural Synchronization

Upon initial execution, the system communicates with the Hugging Face Hub to download and cache the model weights locally. Subsequent boots load directly from the local cache.

```bash
python app.py
```

## Advanced Logic & Heuristics
The prediction pipeline implements several layers of safety and optimization to ensure high-fidelity responses:

Prediction Memory: The system queries the model_corrections collection to prioritize learned manual corrections over raw model inference, allowing for iterative accuracy improvements.

Context Retrieval: It dynamically pulls the last five messages from the user's history to provide a contextualized embedding for the neural engine, ensuring the conversation maintains continuity.

Logic Rules: A secondary heuristic layer via logic_rules.py is applied to handle identity-based queries and provide fixed risk-level assessments for sensitive inputs.