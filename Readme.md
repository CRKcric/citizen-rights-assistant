# Retrieval-Augmented Citizen Rights Interpretation and Grievance Advisory System

A web-based chatbot that helps Indian citizens understand RTI (Right to Information) Act 2005 rules and government grievance procedures (CPGRAMS, DARPG).

**Core principle: no hallucination.** Every answer comes only from retrieved government source text, shows its source citation, and the bot refuses when retrieval confidence is low. It is not a general chatbot.

## Architecture

User question -> retrieval over a vector index of government document chunks -> top chunks passed into a Small Language Model (SLM) prompt -> SLM answers only from those chunks -> answer + citations -> UI.
If the best retrieval score is below a threshold, the SLM is skipped and a refusal is returned.

**Stack:** Phi-3-mini (SLM, via Ollama), BAAI/bge-small-en-v1.5 (embeddings), FAISS (vector store), FastAPI (backend), React + Vite (frontend).

## Repo structure

| Folder | Contents | Owner |
|---|---|---|
| `/data` | Raw documents and chunked JSON | Member A |
| `/retrieval` | Embedding, index, `retrieve()` function | Member A |
| `/backend` | FastAPI, SLM, guardrails | Member B |
| `/frontend` | React app | Member C |

## Team workflow

- Work on separate branches and merge to `main` through pull requests. Never push directly to `main`.
- Commit small and often.
- The interface contract (chunk format, `retrieve()`, `/chat`) changes only by team agreement.

## Interface contract (summary)

**`POST /chat`**
Request: `{"question": "string"}`
Response:
```json
{
  "answer": "string",
  "refused": false,
  "sources": [
    {"source_doc": "RTI Act 2005", "section": "Section 7(1)", "snippet": "short quote", "source_url": "https://..."}
  ]
}
```
When refused: `"refused": true`, `"answer"` holds the refusal message, `"sources": []`.

**`GET /health`** returns `{"status": "ok"}`.
Errors return a proper HTTP status with `{"error": "message"}`.

---

## Backend (Member B)

### Status (Week 1)
- Phi-3-mini runs locally through Ollama.
- FastAPI skeleton with `GET /health` and `POST /chat`.
- `/chat` currently returns a **hardcoded sample response** in the contract shape so the frontend can integrate. Retrieval, the prompt and guardrails come in Weeks 2-3.

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com)

### Set up the model
```
ollama pull phi3:mini
ollama list
```
Ollama must be running in the background whenever the backend or test script calls the model (on Windows and Mac it runs from the system tray; on Linux run `ollama serve`).

Quick check that the model works and how fast it is:
```
python backend/test_phi3.py
```
Observed on first run: about 59 s for the first request (model loading), about 12 s for the next.

### Run the backend
```
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Then open:
- `http://localhost:8000/health` should return `{"status": "ok"}`
- `http://localhost:8000/docs` to try `POST /chat` from the browser

### Notes
- CORS allows the Vite dev server (`http://localhost:5173`). Add the deployed frontend URL in `main.py` when needed.
- Empty questions return HTTP 400 and malformed bodies return HTTP 422, both as `{"error": "..."}`.
- Week 1 finding: with no retrieved context, Phi-3-mini described a "first appeal" as an appeal to a higher court. Under the RTI Act it goes to the First Appellate Authority within the same public authority. This is the baseline that retrieval-grounded answers are compared against.

---

## Knowledge base and retrieval (Member A)

_To be filled by Member A: how to build the index and run `retrieve()`._

## Frontend (Member C)

_To be filled by Member C: how to install and run the React app._