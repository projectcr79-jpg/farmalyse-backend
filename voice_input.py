import os
from openai import OpenAI
import wave
from fastapi import UploadFile, HTTPException
import aiofiles

async def process_voice_input(file: UploadFile):
    # Save the uploaded audio file temporarily
    temp_path = f"temp_{file.filename}"
    async with aiofiles.open(temp_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    # Basic audio validation (assuming WAV for now)
    with wave.open(temp_path, "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2 or wav_file.getframerate() not in [8000, 16000]:
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail="Unsupported audio format. Use mono WAV at 8kHz or 16kHz.")

    # Transcribe using OpenAI Whisper
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    with open(temp_path, "rb") as audio_file:
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )

    # Clean up temporary file
    os.remove(temp_path)

    return transcription