# """
# main.py  —  FastAPI service. Two endpoints: GET /health and POST /chat.
# Run: uvicorn main:app --host 0.0.0.0 --port 8000
# """

# import traceback, sys
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import Literal
# import agent
# from contextlib import asynccontextmanager
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import Literal
# import startup
# import agent

# # import google.generativeai as genai

# app = FastAPI(title="SHL Assessment Recommender")


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     try:
#         startup.build_if_needed()
#     except Exception as e:
#         traceback.print_exc()
#         sys.stdout.flush()
#         raise
#     yield


# app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)


# # ── Schema ────────────────────────────────────────────────────────────────────

# class Message(BaseModel):
#     role: Literal["user", "assistant"]
#     content: str

# class ChatRequest(BaseModel):
#     messages: list[Message]

# class Recommendation(BaseModel):
#     name:      str
#     url:       str
#     test_type: str

# class ChatResponse(BaseModel):
#     reply:               str
#     recommendations:     list[Recommendation]
#     end_of_conversation: bool


# # ── Endpoints ─────────────────────────────────────────────────────────────────

# @app.get("/health")
# def health():
#     return {"status": "ok"}


# @app.post("/chat", response_model=ChatResponse)
# def chat(request: ChatRequest):
#     if len(request.messages) > 8:
#         raise HTTPException(
#             status_code=400,
#             detail="Conversation exceeds 8-turn limit."
#         )
#     try:
#         messages = [{"role": m.role, "content": m.content} for m in request.messages]
#         result   = agent.chat(messages)
#         return ChatResponse(**result)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
    
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import startup
import agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup.build_if_needed()
    agent.init()
    yield


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

class Recommendation(BaseModel):
    name:      str
    url:       str
    test_type: str

class ChatResponse(BaseModel):
    reply:               str
    recommendations:     list[Recommendation]
    end_of_conversation: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if len(request.messages) > 8:
        raise HTTPException(status_code=400, detail="Conversation exceeds 8-turn limit.")
    try:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        result   = agent.chat(messages)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))