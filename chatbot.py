# chatbot.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("C:/Users/chara/cropbot_backend/.env")
openai_api_key = os.getenv("OPENAI_API_KEY")

chatbot_app = FastAPI(title="CropBot AI Chatbot Backend", description="Backend for CropBot AI chatbot using OpenAI", version="1.0")

chatbot_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=openai_api_key)

class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []
    temperature: float = 0.7
    max_tokens: int = 150
    language: str = "en"  # Added for multi-language

@chatbot_app.post("/ask-ai")
async def ask_ai(request: ChatRequest):
    try:
        messages = [
            {"role": "system", "content": f"You are CropBot, an AI assistant specialized in agriculture, farming, plant care, and crop management. Provide helpful, accurate, and concise responses in {request.language}."}
        ]
        messages.extend(request.history)
        messages.append({"role": "user", "content": request.message})

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            presence_penalty=0.6,
        )

        ai_message = response.choices[0].message.content.strip()
        return {
            "status": "success",
            "response": ai_message,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@chatbot_app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "chatbot", "version": "1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(chatbot_app, host="0.0.0.0", port=8001, log_level="info")