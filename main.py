# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
import os
from dotenv import load_dotenv
import requests
import logging
import traceback
import time
from datetime import datetime
from typing import Dict
from openai import OpenAI
import base64
import json
import re
from PIL import Image
import io

# ---- Chatbot mount -------------------------------------------------
from chatbot import chatbot_app

# ---- Logging -------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="CropBot API", version="1.0.0")
app.mount("/chat", chatbot_app)

# ---- CORS ----------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Security ------------------------------------------------------
security = HTTPBearer()
load_dotenv("C:/Users/chara/cropbot_backend/.env")
API_KEY = os.getenv("API_AUTH_KEY", "cropbot-secret-key")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
gemini_api_key = os.getenv("GEMINI_API_KEY")

# ---- Rate-limit & cache -------------------------------------------
remedy_cache: Dict[str, str] = {}
request_count = 0
MAX_REQUESTS_PER_MINUTE = 100
last_reset = time.time()

# ---- File constraints ----------------------------------------------
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024   # 10 MB

# ---- HTML TEST PAGE ------------------------------------------------
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>CropBot AI</title>
    <style>
        body { font-family: Arial; padding: 20px; background: #f4f4f4; }
        .container { max-width: 600px; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        input[type="file"] { margin: 10px 0; }
        button { background: #45c91d; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        pre { background: #f0f0f0; padding: 15px; border-radius: 5px; }
    </style>
</head>
<body>
<div class="container">
    <h2>Upload Crop Image</h2>
    <input type="file" id="fileInput" accept="image/*">
    <button onclick="upload()">Upload and Predict</button>
    <h3>Result:</h3>
    <pre id="result">Waiting...</pre>
</div>

<script>
async function upload() {
    const file = document.getElementById('fileInput').files[0];
    if (!file) return alert("Please select an image");

    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
        headers: { 'Authorization': 'Bearer cropbot-secret-key' }
    });

    const data = await response.json();
    document.getElementById('result').textContent = JSON.stringify(data, null, 2);
}
</script>
</body>
</html>
"""

# --------------------------------------------------------------------
def verify_api_key(api_key: str = Depends(security)):
    if api_key.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    global request_count, last_reset
    now = time.time()
    if now - last_reset > 60:
        request_count = 0
        last_reset = now

    request_count += 1
    if request_count > MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    return api_key

# --------------------------------------------------------------------
def validate_file(file: UploadFile, content: bytes | None = None):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    if content and len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10 MB)")

# --------------------------------------------------------------------
def compress_image(file_content: bytes) -> bytes:
    """Compress image to 512x512 max size to reduce Gemini upload time."""
    try:
        img = Image.open(io.BytesIO(file_content))
        img = img.convert("RGB")
        img.thumbnail((512, 512))   # Resize

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()
    except Exception:
        return file_content  # fallback

# --------------------------------------------------------------------
def call_gemini_api(file_content: bytes, filename: str) -> Dict:
    """Gemini 2.5 Flash detection with retry + timeout + safer JSON extraction."""

    b64 = base64.b64encode(file_content).decode("utf-8")

    prompt = (
        "Analyze this plant leaf image. Return ONLY JSON with:\n"
        "- scientific_name\n"
        "- common_name\n"
        "- disease ('None' if healthy)\n"
        "- confidence (0-100)\n"
        "- causes (2-3 causes)\n"
    )

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={gemini_api_key}"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                ]
            }
        ]
    }

    # ---- Retry up to 3 times ----
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, timeout=50)
            response.raise_for_status()

            result = response.json()
            logger.info(f"Gemini result: {result}")

            text_part = result["candidates"][0]["content"]["parts"][0]["text"]

            json_match = re.search(r"\{.*\}", text_part, re.DOTALL)
            if not json_match:
                return {
                    "scientific_name": "Unknown",
                    "common_name": "Unknown",
                    "disease": "None",
                    "confidence": 0.0,
                    "causes": []
                }

            data = json.loads(json_match.group())
            return {
                "scientific_name": data.get("scientific_name", "Unknown"),
                "common_name": data.get("common_name", "Unknown"),
                "disease": data.get("disease", "None"),
                "confidence": float(data.get("confidence", 0.0)),
                "causes": data.get("causes", [])
            }

        except requests.exceptions.Timeout:
            logger.warning(f"Gemini timeout retry {attempt + 1}/3")
            if attempt == 2:
                raise HTTPException(503, "Gemini API timed out")

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            if attempt == 2:
                raise HTTPException(503, "Gemini service unavailable")
            time.sleep(1)

# --------------------------------------------------------------------
def get_remedy_from_openai(scientific_name: str, common_name: str, disease: str, causes: list) -> str:
    """Generate remedy text using OpenAI."""

    key = f"{scientific_name}_{disease}".lower()
    if key in remedy_cache:
        return remedy_cache[key]

    causes_str = "\n".join([f"- {c}" for c in causes]) or "- No causes detected"

    prompt = (
        f"Plant: {common_name} ({scientific_name})\n"
        f"Disease: {disease}\n"
        f"Causes:\n{causes_str}\n\n"
        "Provide remedy in English using:\n"
        "**Disease Name:**\n"
        "**Causes:**\n"
        "**Natural Solutions:**\n"
        "**Chemical Solutions:**\n"
        "**Care Tips:**\n"
        "Keep under 180 tokens."
    )

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a plant disease expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=180,
            temperature=0.3
        )
        remedy = resp.choices[0].message.content.strip()
        remedy_cache[key] = remedy
        return remedy

    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return (
            f"Disease Name: {disease}\n"
            f"Causes:\n{causes_str}\n"
            "Natural Solutions: - Remove infected leaves\n"
            "Chemical Solutions: - Use recommended fungicide\n"
            "Care Tips: - Improve air flow"
        )

# --------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=html_template)

# --------------------------------------------------------------------
@app.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    start = time.time()
    try:
        content = await file.read()

        # ---- Compress image (major speed improvement) ----
        content = compress_image(content)

        validate_file(file, content)

        # Step 1: Gemini analysis
        gemini_result = call_gemini_api(content, file.filename)

        sci_name = gemini_result["scientific_name"]
        com_name = gemini_result["common_name"]
        disease = gemini_result["disease"]
        confidence = gemini_result["confidence"]
        causes = gemini_result["causes"]

        # Step 2: OpenAI remedy
        remedy = get_remedy_from_openai(sci_name, com_name, disease, causes)

        return JSONResponse({
            "status": "success",
            "scientific_name": sci_name,
            "common_name": com_name,
            "confidence": round(confidence, 1),
            "disease": disease,
            "causes": causes,
            "recommendations": remedy,
            "time": round(time.time() - start, 2)
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}\n{traceback.format_exc()}")
        return JSONResponse({
            "status": "error",
            "message": "Internal server error",
            "time": round(time.time() - start, 2)
        })

# --------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "cropbot", "time": datetime.now().isoformat()}

# --------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting CropBot on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
