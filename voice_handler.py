import logging
import os
from fastapi import APIRouter, UploadFile, HTTPException
from openai import OpenAI
from dotenv import load_dotenv

# Initialize
router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv("C:/Users/chara/cropbot_backend/.env")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@router.post("/voice-ask-ai")
async def voice_ask_ai(file: UploadFile):
    try:
        # Save uploaded audio temporarily
        temp_path = os.path.join(os.path.dirname(__file__), "temp_audio.wav")
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        logger.info(f"Audio file saved to {temp_path}")

        # Step 1: Transcribe using Whisper API
        with open(temp_path, "rb") as audio_file:
            transcription = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        query_text = transcription.text.strip()
        logger.info(f"Transcribed text: {query_text}")

        if not query_text:
            raise HTTPException(status_code=400, detail="No speech detected in audio")

        # Step 2: Get AI response from ChatGPT
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are CropBot, an AI farming assistant."},
                {"role": "user", "content": query_text},
            ],
            temperature=0.7,
            max_tokens=150
        )

        ai_reply = response.choices[0].message.content.strip()
        logger.info(f"AI reply: {ai_reply}")

        return {"response": ai_reply, "transcription": query_text}

    except Exception as e:
        logger.error(f"Voice processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
