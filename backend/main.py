"""Citizen Rights Assistant - backend (Member B, Week 2).

/chat now runs: retrieve -> prompt Phi-3-mini -> answer + citations.
Uses Member A's retriever from ../retrieval (falls back to the stub, loudly).
Guards (score threshold, NO_ANSWER check) live in rag.answer_question().

Run:  uvicorn main:app --reload --port 8000
"""
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rag import ModelTimeoutError, ModelUnavailableError, answer_question

# Member A's real retriever lives in ../retrieval. Fall back to the stub loudly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
try:
    from retrieve import retrieve
    print("Using REAL retriever (Member A)")
except ImportError as exc:
    from stub_retriever import retrieve
    print(f"!!! USING STUB RETRIEVER ({exc}) !!!")

app = FastAPI(title="Citizen Rights Assistant API")

# CORS: allow the Vite dev server (add the deployed frontend URL later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


# Contract: errors return {"error": "message"} with a proper HTTP status.
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"error": 'Request body must be {"question": "string"}'},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        return JSONResponse(
            status_code=400, content={"error": "Question must not be empty"}
        )

    try:
        chunks = retrieve(question, top_k=4)
        return answer_question(question, chunks)
    except ModelUnavailableError:
        return JSONResponse(
            status_code=503,
            content={"error": "The language model is not running. Please try again later."},
        )
    except ModelTimeoutError:
        return JSONResponse(
            status_code=504,
            content={"error": "The language model took too long to respond."},
        )