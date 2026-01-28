import os
import io
import zipfile
import tempfile
import shutil
from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import JSONResponse


FLAG = os.environ.get("FLAG", "FLAG{example_flag_change_me}")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))
SUBMIT_TOKEN = os.environ.get("SUBMIT_TOKEN", "")

MAX_ZIP_BYTES = int(os.environ.get("MAX_ZIP_BYTES", str(50 * 1024 * 1024)))          # 50MB
MAX_EXTRACTED_BYTES = int(os.environ.get("MAX_EXTRACTED_BYTES", str(200 * 1024 * 1024)))  # 200MB

# 30 hidden tests (keep private on server)
HIDDEN_TEST_SET: List[Tuple[str, int]] = [
    ("Congratulations! You have won a free iPhone. Click now to claim.", 1),
    ("Hey team, meeting moved to 15:00. Please confirm attendance.", 0),
    ("URGENT: Your account will be suspended. Verify your password here.", 1),
    ("Can you review the attached quarterly report before tomorrow?", 0),
    ("Limited time offer!!! Earn money fast from home, sign up today.", 1),
    ("Would you like to schedule a lunch meeting sometime next week? Let me know what works for you.", 0),
    ("You have been selected for a prize. Provide details to receive it.", 1),
    ("Just a note that expense reports are due by the end of the week.", 0),
    ("Act now to unlock exclusive benefits. Click the link.", 1),
    ("Please find the updated project plan in the shared folder.", 0),
    ("Final notice: invoice overdue. Pay immediately to avoid fees.", 1),
    ("Thanks for your help today — much appreciated.", 0),
    ("Get cheap meds now, no prescription needed, confidential shipping.", 1),
    ("Could you send me the minutes from yesterday's call?", 0),
    ("Your mailbox storage is full. Increase quota by logging in.", 1),
    ("FYI: the deployment window is scheduled for Saturday morning.", 0),
    ("Win cash prizes instantly — reply YES to participate!", 1),
    ("Please approve the access request in the ticketing system.", 0),
    ("Exclusive deal: 90% discount, only today!", 1),
    ("I will be out of the office tomorrow. I will respond as soon as I’m back.", 0),
    ("Security alert: suspicious login detected. Confirm your identity.", 1),
    ("Can we reschedule our 1:1 to Wednesday?", 0),
    ("Claim your reward now. Verification required.", 1),
    ("Please find the agenda for today’s workshop attached for your reference.", 0),
    ("Hot singles in your area waiting for you. Click here.", 1),
    ("Invoice attached. Let me know if any questions.", 0),
    ("Earn $$$ fast, no experience needed. Start today.", 1),
    ("Please update your availability in the calendar.", 0),
    ("Your payment method failed. Update details to continue service.", 1),
    ("Great job on the presentation — the customer loved it.", 0),
]


# If students saved with a custom TransformerBlock, define it here for loading .keras
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim=10, num_heads=2, ff_dim=32, rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        key_dim = max(1, embed_dim // max(1, num_heads))
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn = self.att(inputs, inputs, training=training)
        attn = self.dropout1(attn, training=training)
        out1 = self.layernorm1(inputs + attn)
        ffn = self.ffn(out1, training=training)
        ffn = self.dropout2(ffn, training=training)
        return self.layernorm2(out1 + ffn)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate})
        return cfg


CUSTOM_OBJECTS = {"TransformerBlock": TransformerBlock}

app = FastAPI(title="Student Validator", version="1.0")


def safe_extract_zip(zip_bytes: bytes, dest_dir: str) -> None:
    if len(zip_bytes) > MAX_ZIP_BYTES:
        raise HTTPException(status_code=400, detail=f"Zip too large (>{MAX_ZIP_BYTES} bytes).")

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        total = 0
        for info in zf.infolist():
            # zip slip protection
            extracted_path = os.path.normpath(os.path.join(dest_dir, info.filename))
            if not extracted_path.startswith(os.path.abspath(dest_dir)):
                raise HTTPException(status_code=400, detail="Unsafe zip content (path traversal).")

            total += info.file_size
            if total > MAX_EXTRACTED_BYTES:
                raise HTTPException(status_code=400, detail="Extracted content too large (zip bomb protection).")

        zf.extractall(dest_dir)


def find_model(extracted_dir: str) -> Dict[str, str]:
    # Look for .keras
    for root, _, files in os.walk(extracted_dir):
        for f in files:
            if f.endswith(".keras"):
                return {"type": "keras", "path": os.path.join(root, f)}
    # Look for SavedModel
    for root, _, files in os.walk(extracted_dir):
        if "saved_model.pb" in files:
            return {"type": "savedmodel", "path": root}
    raise HTTPException(status_code=400, detail="No model found (.keras or SavedModel with saved_model.pb).")


def load_any_model(model_info: Dict[str, str]):
    if model_info["type"] == "keras":
        return keras.models.load_model(model_info["path"], custom_objects=CUSTOM_OBJECTS, compile=False)
    if model_info["type"] == "savedmodel":
        return keras.models.load_model(model_info["path"], custom_objects=CUSTOM_OBJECTS, compile=False)
    raise HTTPException(status_code=400, detail="Unsupported model type.")


def predict_probs(model: keras.Model, texts: List[str]) -> np.ndarray:
    # Require models that accept raw strings (recommended workshop contract)
    preds = model.predict(np.array(texts, dtype=object), verbose=0)
    return preds.reshape(-1).astype(float)


@app.get("/")
def home():
    return {"status": "ok", "info": "POST a zip to /submit with header X-Submit-Token."}


@app.post("/submit")
async def submit(file: UploadFile = File(...), x_submit_token: str = Header(default="")):
    if SUBMIT_TOKEN and x_submit_token != SUBMIT_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid submission token.")

    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip file.")

    zip_bytes = await file.read()

    tmpdir = tempfile.mkdtemp(prefix="submit_")
    try:
        safe_extract_zip(zip_bytes, tmpdir)
        model_info = find_model(tmpdir)
        model = load_any_model(model_info)

        texts = [t for t, _ in HIDDEN_TEST_SET]
        labels = np.array([y for _, y in HIDDEN_TEST_SET], dtype=int)

        probs = predict_probs(model, texts)
        preds = (probs >= THRESHOLD).astype(int)

        acc = float((preds == labels).mean())
        false_positives = int(((labels == 0) & (preds == 1)).sum())
        false_negatives = int(((labels == 1) & (preds == 0)).sum())

        passed = (acc > 0.98) and (false_positives == 0)

        metrics = {
            "detection_rate": acc,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "threshold": THRESHOLD,
            "model_type": model_info["type"],
        }

        if passed:
            return JSONResponse({"result": "PASS", "flag": FLAG, "metrics": metrics})

        return JSONResponse(
            {
                "result": "FAIL",
                "metrics": metrics,
                "hint": "Tune hyperparams, clean text, reduce false positives. Consider threshold tuning & overfitting control.",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation failed: {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
