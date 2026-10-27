"""Citizen Rights Assistant - backend (Member B, Week 2).

/chat now runs: retrieve -> prompt Phi-3-mini -> answer + citations.
Retrieval is still the STUB. When Member A's retrieve() is ready, change
the import below (one line). Refusal threshold and guardrails come in Week 3.

Run:  uvicorn main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rag import ModelTimeoutError, ModelUnavailableError, answer_question
from stub_retriever import retrieve  # TODO: swap for Member A's retrieve()

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