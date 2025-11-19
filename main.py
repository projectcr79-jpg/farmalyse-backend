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

# -------------------- Configuration & Env --------------------
load_dotenv()  # loads .env if present

API_KEY = os.getenv("API_AUTH_KEY", "cropbot-secret-key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Optional OpenAI client
if OPENAI_API_KEY and OpenAI:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None

# Gemini endpoint
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# Rate limit config
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "120"))

# File constraints
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# -------------------- Logging & App --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cropbot")

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

security = HTTPBearer()

# In-memory counters/caches (process-level)
request_count = 0
last_reset = time.time()
remedy_cache: Dict[str, str] = {}

# -------------------- Small helpers --------------------
def strip_code_fences_and_control_chars(text: str) -> str:
    """
    Remove common markdown code fences and non-printable/control characters that break json.loads.
    Keeps newline and horizontal whitespace.
    """
    if not isinstance(text, str):
        return text
    # Remove triple backtick fences and language hints (e.g., ```json)
    text = re.sub(r"```(?:\w+)?\s*", "", text)
    text = text.replace("```", "")
    # Remove stray single backticks
    text = text.replace("`", "")
    # Remove BOM and other control chars except newline / tab
    cleaned = "".join(ch for ch in text if (ord(ch) >= 32) or ch in ("\n", "\t"))
    # Strip leading/trailing whitespace
    return cleaned.strip()

def _extract_first_json_object(text: str) -> Optional[dict]:
    """
    Extract the first {...} JSON object from text. Return dict or None.
    Uses a safe greedy fallback and strips control chars first.
    """
    if not text:
        return None
    text = strip_code_fences_and_control_chars(text)
    # Find the first balanced JSON object using a regex that works for typical outputs
    match = re.search(r"\{(?:[^{}]|(?R))*\}", text, re.DOTALL) if hasattr(re, "search") else re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        # fallback: find braces with simple greedy
        match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    candidate = match.group()
    try:
        return json.loads(candidate)
    except Exception as e:
        logger.debug(f"JSON parse failed after cleaning: {e}. Raw candidate: {candidate}")
        # As a last resort, attempt to replace smart quotes and re-try
        normalized = candidate.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
        try:
            return json.loads(normalized)
        except Exception as ee:
            logger.warning(f"Final JSON parse failed: {ee}")
            return None

def compress_image(file_content: bytes, max_size: int = 512) -> bytes:
    """Compress image to JPEG thumbnail to reduce upload size/time."""
    try:
        img = Image.open(io.BytesIO(file_content))
        img = img.convert("RGB")
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Image compression failed: {e}")
        return file_content

def validate_file(file: UploadFile, content: Optional[bytes] = None):
    """Validate file presence, extension, and size."""
    if not file or not getattr(file, "filename", None):
        raise HTTPException(status_code=400, detail="No file uploaded")
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    if content and len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10 MB)")

# -------------------- Gemini call --------------------
def call_gemini_api(file_content: bytes, filename: str, max_retries: int = 4) -> Dict:
    """
    Call Gemini 2.5-flash to get structured plant disease JSON.
    - Uses correct camelCase keys: inlineData, mimeType
    - Forces JSON-only response in prompt
    - Cleans fences, control chars, retries on transient errors (503/timeouts)
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key missing")

    b64 = base64.b64encode(file_content).decode("utf-8")

    # Strict prompt that asks for only JSON and an exact structure
    prompt = (
        "You are a plant disease classifier. Analyze the attached image and "
        "RETURN ONLY a single JSON object, with NO MARKDOWN, NO EXPLANATION, NO TRIPLE BACKTICKS.\n\n"
        "The JSON MUST exactly include these fields:\n"
        "scientific_name (string), common_name (string), disease (string or 'None'), "
        "confidence (number 0-100), causes (array of 2-3 strings).\n\n"
        "Example:\n"
        "{\"scientific_name\":\"Solanum lycopersicum\",\"common_name\":\"Tomato\",\"disease\":\"Late Blight\",\"confidence\":94.5,\"causes\":[\"High humidity\",\"Fungal spores\"]}\n\n"
        "Return only the JSON object."
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": b64
                        }
                    }
                ]
            }
        ],
        # generationConfig optional tweaks can be added here
        "generationConfig": {
            "temperature": 0.0,
            # You can add other fields like maxOutputTokens or thinkingBudget if desired
        }
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    backoff = 1.0
    last_text = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(GEMINI_URL, json=payload, headers=headers, timeout=40)
            # If overloaded, retry with backoff
            if resp.status_code == 503:
                logger.warning(f"Gemini overloaded (503). Attempt {attempt}/{max_retries}. Backoff {backoff}s.")
                time.sleep(backoff)
                backoff *= 2
                continue

            resp.raise_for_status()
            j = resp.json()
            logger.debug("Gemini raw response: %s", j)

            # Try multiple possible response shapes to extract text
            text_out = ""
            try:
                candidates = j.get("candidates")
                if candidates and isinstance(candidates, list) and len(candidates) > 0:
                    cand = candidates[0]
                    content = cand.get("content") or {}
                    parts = content.get("parts") or []
                    for p in parts:
                        if isinstance(p, dict) and "text" in p and p.get("text"):
                            text_out += p.get("text", "")
                        elif isinstance(p, dict) and "content" in p and isinstance(p["content"], str):
                            text_out += p["content"]
            except Exception:
                text_out = ""

            # fallback fields
            if not text_out:
                text_out = j.get("output_text") or j.get("text") or json.dumps(j)

            last_text = text_out

            # Clean text and extract JSON
            parsed = _extract_first_json_object(text_out)
            if parsed:
                # Normalize and return
                return {
                    "scientific_name": parsed.get("scientific_name", "Unknown"),
                    "common_name": parsed.get("common_name", "Unknown"),
                    "disease": parsed.get("disease", "None"),
                    "confidence": float(parsed.get("confidence", 0.0)),
                    "causes": parsed.get("causes", []) if isinstance(parsed.get("causes", []), list) else []
                }
            else:
                logger.warning("Gemini returned no parsable JSON. Attempt %s/%s. Raw text: %s", attempt, max_retries, strip_code_fences_and_control_chars(text_out))
                if attempt == max_retries:
                    # Final fallback to Unknown
                    return {
                        "scientific_name": "Unknown",
                        "common_name": "Unknown",
                        "disease": "None",
                        "confidence": 0.0,
                        "causes": []
                    }
                time.sleep(backoff)
                backoff *= 2
                continue

        except requests.exceptions.Timeout:
            logger.warning("Gemini request timed out. Attempt %s/%s. Backoff %ss", attempt, max_retries, backoff)
            if attempt == max_retries:
                raise HTTPException(status_code=503, detail="Gemini timed out")
            time.sleep(backoff)
            backoff *= 2
            continue
        except requests.exceptions.HTTPError as he:
            logger.error("Gemini HTTP error: %s - raw: %s", he, getattr(resp, "text", None))
            if attempt == max_retries:
                raise HTTPException(status_code=503, detail="Gemini service unavailable")
            time.sleep(backoff)
            backoff *= 2
            continue
        except Exception as e:
            logger.error("Unexpected Gemini error: %s - last resp text: %s", e, strip_code_fences_and_control_chars(last_text) if last_text else None)
            if attempt == max_retries:
                raise HTTPException(status_code=503, detail="Gemini service unavailable")
            time.sleep(backoff)
            backoff *= 2
            continue

    # Shouldn't reach here
    raise HTTPException(status_code=503, detail="Gemini failed after retries")

# -------------------- OpenAI remedy generator (fallback/cache) --------------------
def get_remedy_from_openai(scientific_name: str, common_name: str, disease: str, causes: List[str]) -> str:
    """
    Generate a short remedy using OpenAI. If OpenAI not configured, return a conservative fallback string.
    """
    key = f"{scientific_name}_{disease}".lower()
    if key in remedy_cache:
        return remedy_cache[key]

    causes_str = "\n".join([f"- {c}" for c in causes]) if causes else "- No causes detected"

    prompt = (
        f"Plant: {common_name} ({scientific_name})\n"
        f"Disease: {disease}\n"
        f"Causes:\n{causes_str}\n\n"
        "Provide a concise remedy in plain English using the sections:\n"
        "Disease Name:\nCauses:\nNatural Solutions:\nChemical Solutions:\nCare Tips:\nKeep it short (about 120-200 tokens)."
    )

    # If OpenAI client not configured, return simple fallback
    if not openai_client:
        fallback = (
            f"Disease Name: {disease}\n"
            f"Causes:\n{causes_str}\n"
            "Natural Solutions: - Remove infected tissue, increase airflow, use neem oil\n"
            "Chemical Solutions: - Apply recommended pesticide/fungicide per label\n"
            "Care Tips: - Avoid overhead watering; monitor crop"
        )
        remedy_cache[key] = fallback
        return fallback

    try:
        resp = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a helpful plant disease agronomist."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=220,
            temperature=0.2
        )
        remedy = resp.choices[0].message.content.strip()
        remedy_cache[key] = remedy
        return remedy
    except Exception as e:
        logger.error("OpenAI error: %s", e)
        fallback = (
            f"Disease Name: {disease}\n"
            f"Causes:\n{causes_str}\n"
            "Natural Solutions: - Remove infected leaves, improve airflow\n"
            "Chemical Solutions: - Use appropriate fungicide\n"
            "Care Tips: - Avoid overhead watering"
        )
        remedy_cache[key] = fallback
        return fallback

# -------------------- Auth middleware --------------------
def verify_api_key(api_key: str = Depends(security)):
    """Verify Bearer API key and basic per-process rate limit."""
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

# -------------------- Routes --------------------
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

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=html_template)

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    start = time.time()
    try:
        content = await file.read()
        validate_file(file, content)

        # compress to speed up uploads
        content = compress_image(content, max_size=512)

        # call Gemini -> returns normalized dict
        gemini_result = call_gemini_api(content, file.filename)

        sci_name = gemini_result.get("scientific_name", "Unknown")
        com_name = gemini_result.get("common_name", "Unknown")
        disease = gemini_result.get("disease", "None")
        confidence = float(gemini_result.get("confidence", 0.0))
        causes = gemini_result.get("causes", []) or []

        # generate remedy using OpenAI (or fallback)
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
        logger.error("Prediction error: %s\n%s", e, traceback.format_exc())
        return JSONResponse({
            "status": "error",
            "message": "Internal server error",
            "error": str(e),
            "time_seconds": round(time.time() - start, 2)
        }, status_code=500)

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "cropbot", "time": datetime.utcnow().isoformat()}

# -------------------- CLI runner --------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting CropBot on 0.0.0.0:%s", os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")
