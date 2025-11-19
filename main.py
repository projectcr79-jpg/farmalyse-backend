# main.py
import os
import io
import re
import time
import json
import base64
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Optional

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from dotenv import load_dotenv
from PIL import Image

# Optional OpenAI client import (your code used OpenAI)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Try to import chatbot_app if available (safe)
try:
    from chatbot import chatbot_app  # type: ignore
    HAVE_CHATBOT = True
except Exception:
    HAVE_CHATBOT = False

# ---- Load environment ----------------------------------------------------
load_dotenv()  # loads .env from cwd if present

API_KEY = os.getenv("API_AUTH_KEY", "cropbot-secret-key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if OPENAI_API_KEY and OpenAI:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None

# ---- Logging -------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("cropbot")

# ---- App + CORS ---------------------------------------------------------
app = FastAPI(title="CropBot API", version="1.0.0")

if HAVE_CHATBOT:
    app.mount("/chat", chatbot_app)
    logger.info("Mounted chatbot_app at /chat")
else:
    logger.info("chatbot_app not found; skipping mount")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Security + Rate limiting -------------------------------------------
security = HTTPBearer()

# Very small in-memory rate limiter (process-level)
request_count = 0
last_reset = time.time()
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "100"))

# Simple remedy cache in-memory
remedy_cache: Dict[str, str] = {}

# ---- File constraints ---------------------------------------------------
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# ---- HTML test page ----------------------------------------------------
html_template = """
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>CropBot AI</title></head>
<body>
  <h2>CropBot AI - Upload</h2>
  <input type="file" id="fileInput" accept="image/*" />
  <button onclick="upload()">Upload</button>
  <pre id="out">Waiting...</pre>
  <script>
  async function upload() {
    const f = document.getElementById('fileInput').files[0];
    if(!f) return alert("choose file");
    const fd = new FormData();
    fd.append('file', f);
    const res = await fetch('/predict', {
      method: 'POST',
      body: fd,
      headers: {'Authorization': 'Bearer ' + '""" + API_KEY + """'}
    });
    const j = await res.json();
    document.getElementById('out').innerText = JSON.stringify(j, null, 2);
  }
  </script>
</body>
</html>
"""

# -------------------------------------------------------------------------
def verify_api_key(api_key: str = Depends(security)):
    """Verify Bearer API key and do a trivial rate limit."""
    global request_count, last_reset
    if api_key.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    now = time.time()
    if now - last_reset > 60:
        request_count = 0
        last_reset = now

    request_count += 1
    if request_count > MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    return api_key


def validate_file(file: UploadFile, content: Optional[bytes] = None):
    """Validate filename, extension and size."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    if content and len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10 MB)")


def compress_image(file_content: bytes, max_size: int = 512) -> bytes:
    """
    Compress/resize image to reduce upload size while retaining readable detail.
    Returns JPEG bytes.
    """
    try:
        img = Image.open(io.BytesIO(file_content))
        img = img.convert("RGB")
        img.thumbnail((max_size, max_size))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()
    except Exception as e:
        logger.warning(f"Image compress failed, returning original: {e}")
        return file_content


def _extract_json_from_text(text: str) -> Optional[dict]:
    """Find first JSON object in text and return parsed dict, or None."""
    match = re.search(r"\{(?:[^{}]|(?R))*\}", text, re.DOTALL) if hasattr(re, "search") else re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        # fallback simple greedy
        match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception as e:
        logger.warning(f"Failed to parse JSON from Gemini text: {e}")
        return None


def call_gemini_api(file_content: bytes, filename: str) -> Dict:
    """
    Call Gemini 2.5 Flash model to analyze the image and return normalized JSON.
    Uses x-goog-api-key header for authentication (API key).
    Retries up to 3 times on transient failures.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")

    b64 = base64.b64encode(file_content).decode("utf-8")

    prompt = (
        "You are a plant disease classifier. Analyze the attached leaf image and "
        "return ONLY a JSON object with these keys:\n"
        "scientific_name, common_name, disease (or 'None'), confidence (0-100), causes (array of 2-3 strings).\n"
        "Example:\n"
        "{\"scientific_name\":\"Solanum lycopersicum\",\"common_name\":\"Tomato\",\"disease\":\"Late Blight\",\"confidence\":94.5,\"causes\":[\"High humidity\",\"Fungal spores\"]}"
    )

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64
                        }
                    }
                ]
            }
        ],
        # Optionally add generationConfig here if needed
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    last_resp_text = None
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=40)
            resp.raise_for_status()
            result = resp.json()
            logger.debug(f"Gemini raw response: {result}")

            # Extract content text safely
            text_out = ""
            # primary expected path
            try:
                candidate = (result.get("candidates") or [None])[0]
                if candidate:
                    content = candidate.get("content") or {}
                    parts = content.get("parts") or []
                    for p in parts:
                        if isinstance(p, dict) and "text" in p:
                            text_out += p.get("text", "")
                        elif isinstance(p, dict) and "content" in p and isinstance(p["content"], str):
                            text_out += p["content"]
            except Exception:
                text_out = ""

            # fallback common keys
            if not text_out:
                text_out = result.get("output_text") or result.get("text") or json.dumps(result)

            last_resp_text = text_out

            data = _extract_json_from_text(text_out)
            if not data:
                logger.warning("Gemini did not return a parsable JSON object in text.")
                # treat as transient / malformed -> retry (unless last attempt)
                if attempt == 2:
                    # final fallback: return safe unknown
                    return {
                        "scientific_name": "Unknown",
                        "common_name": "Unknown",
                        "disease": "None",
                        "confidence": 0.0,
                        "causes": []
                    }
                time.sleep(1)
                continue

            # Normalize fields
            return {
                "scientific_name": data.get("scientific_name", "Unknown"),
                "common_name": data.get("common_name", "Unknown"),
                "disease": data.get("disease", "None"),
                "confidence": float(data.get("confidence", 0.0)),
                "causes": data.get("causes", []) if isinstance(data.get("causes", []), list) else []
            }

        except requests.exceptions.Timeout:
            logger.warning(f"Gemini timeout (attempt {attempt+1}/3)")
            if attempt == 2:
                raise HTTPException(status_code=503, detail="Gemini API timed out")
            time.sleep(1)
        except requests.exceptions.HTTPError as he:
            logger.error(f"Gemini HTTP error: {he} - raw: {getattr(resp, 'text', None)}")
            if attempt == 2:
                raise HTTPException(status_code=503, detail="Gemini service unavailable")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Unexpected Gemini error: {e} - last resp text: {last_resp_text}")
            if attempt == 2:
                raise HTTPException(status_code=503, detail="Gemini service unavailable")
            time.sleep(1)

    # ultimate fallback
    return {
        "scientific_name": "Unknown",
        "common_name": "Unknown",
        "disease": "None",
        "confidence": 0.0,
        "causes": []
    }


def get_remedy_from_openai(scientific_name: str, common_name: str, disease: str, causes: List[str]) -> str:
    """Generate remedy using OpenAI Chat completions (cached)."""
    key = f"{scientific_name}_{disease}".lower()
    if key in remedy_cache:
        return remedy_cache[key]

    causes_str = "\n".join([f"- {c}" for c in causes]) or "- No causes detected"

    prompt = (
        f"Plant: {common_name} ({scientific_name})\n"
        f"Disease: {disease}\n"
        f"Causes:\n{causes_str}\n\n"
        "Provide a concise remedy in English using the following sections:\n"
        "**Disease Name:**\n"
        "**Causes:**\n"
        "**Natural Solutions:**\n"
        "**Chemical Solutions:**\n"
        "**Care Tips:**\n"
        "Keep it short (around 150-200 tokens)."
    )

    if not openai_client:
        # Conservative fallback if OpenAI client isn't available
        short = (
            f"Disease Name: {disease}\n"
            f"Causes:\n{causes_str}\n"
            "Natural Solutions: - Remove infected leaves, improve airflow\n"
            "Chemical Solutions: - Use appropriate fungicide per crop label\n"
            "Care Tips: - Avoid overhead watering"
        )
        return short

    try:
        resp = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a plant disease expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=220,
            temperature=0.2
        )
        remedy_text = resp.choices[0].message.content.strip()
        remedy_cache[key] = remedy_text
        return remedy_text
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        # fallback short remedy
        return (
            f"Disease Name: {disease}\n"
            f"Causes:\n{causes_str}\n"
            "Natural Solutions: - Remove infected leaves\n"
            "Chemical Solutions: - Use recommended fungicide\n"
            "Care Tips: - Improve air flow"
        )


# ---- Routes --------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=html_template)


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    start = time.time()
    try:
        content = await file.read()
        validate_file(file, content)

        # compress (optional)
        content = compress_image(content, max_size=512)

        # call Gemini
        gemini_result = call_gemini_api(content, file.filename)

        sci_name = gemini_result.get("scientific_name", "Unknown")
        com_name = gemini_result.get("common_name", "Unknown")
        disease = gemini_result.get("disease", "None")
        confidence = float(gemini_result.get("confidence", 0.0))
        causes = gemini_result.get("causes", []) or []

        # get remedy
        remedy = get_remedy_from_openai(sci_name, com_name, disease, causes)

        response = {
            "status": "success",
            "scientific_name": sci_name,
            "common_name": com_name,
            "disease": disease,
            "confidence": round(confidence, 1),
            "causes": causes,
            "recommendations": remedy,
            "time_seconds": round(time.time() - start, 2)
        }
        return JSONResponse(response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}\n{traceback.format_exc()}")
        return JSONResponse({
            "status": "error",
            "message": "Internal server error",
            "error": str(e),
            "time_seconds": round(time.time() - start, 2)
        }, status_code=500)


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "cropbot", "time": datetime.utcnow().isoformat()}


# ---- CLI run -------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting CropBot on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")
