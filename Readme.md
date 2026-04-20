# Chandrakanth K — AI Portfolio Chatbot

A Gradio-based personal portfolio chatbot powered by GPT-4o-mini, RAG (Retrieval-Augmented Generation) using pgvector on Neon PostgreSQL, with an admin panel for managing unanswered questions. Deployed on Hugging Face Spaces.

---

## Features

- **AI Chatbot** — Answers questions about Chandrakanth's skills, experience, and projects on his behalf using a custom persona system prompt
- **RAG Pipeline** — Retrieves relevant context from uploaded documents (resume, project docs, etc.) using vector similarity search
- **Streaming Responses** — GPT-4o-mini responses stream token by token in the UI
- **Chat History / Memory** — Maintains conversation context within a session
- **Document Upload** — Pre-load docs at startup + drag-and-drop upload at runtime via Gradio UI
- **Unanswered Question Logging** — Questions the bot cannot confidently answer are saved to PostgreSQL with `status = pending`
- **Admin Tab** — In-app Gradio tab to view pending questions, add answers, and manage the Q&A store
- **Human-in-the-Loop** — Once you add an answer in the admin tab, it is reused automatically when the same (or similar) question is asked again
- **Resume Download Button** — One-click resume download from the chat UI
- **Contact Link** — Direct link to LinkedIn profile

---

## Project Structure

```
chandrakanth-portfolio-chatbot/
│
├── app.py                  # Main Gradio app — UI layout, tab wiring, event handlers
├── chatbot.py              # Core chat logic — query router, answer builder, streaming
├── rag.py                  # Document ingestion, chunking, embedding, vector search
├── db.py                   # All PostgreSQL operations via psycopg2
├── admin.py                # Admin tab helpers — load/update/delete Q&A pairs
├── persona.py              # System prompt definition (Chandrakanth's persona)
│
├── docs/                   # Pre-loaded documents (resume, project notes, etc.)
│   └── Chandrakanth_K_Resume.pdf
│
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (never commit this)
├── .env.example            # Example env file to share safely
│
└── README.md
```

---

## Architecture Overview

```
User Question
     │
     ▼
Query Router
     │
     ├──► Q&A Store (PostgreSQL)       ← exact/similar match from human-answered pairs
     ├──► Doc Vectors (pgvector)       ← semantic search over uploaded documents
     └──► GPT-4o-mini                  ← persona-aware LLM with RAG context injected
                │
                ▼
        Answer Builder
        (merge all context → stream reply)
                │
                ▼
        Confidence Check
         /            \
        YES            NO
         │              │
    Stream to        Log to PostgreSQL
      User           (status = pending)
                          │
                    You add answer
                    in Admin Tab
                          │
                    Feeds back into
                      Q&A Store
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI Framework | Gradio 4.x |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | pgvector extension on Neon PostgreSQL |
| Database | Neon (serverless PostgreSQL, free tier) |
| Document Parsing | PyMuPDF (PDF), python-docx (DOCX), plain text |
| Hosting | Hugging Face Spaces |
| Language | Python 3.10+ |

---

## PostgreSQL Schema

Run these SQL statements in your Neon console to set up the database.

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Table: document chunks with embeddings
CREATE TABLE IF NOT EXISTS document_chunks (
    id          SERIAL PRIMARY KEY,
    source      TEXT NOT NULL,              -- filename
    chunk_index INTEGER NOT NULL,           -- chunk number within file
    content     TEXT NOT NULL,              -- raw text chunk
    embedding   vector(1536),              -- OpenAI text-embedding-3-small dimension
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Index for fast vector similarity search
CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding
ON document_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Table: Q&A store (human-answered pairs + unanswered log)
CREATE TABLE IF NOT EXISTS qa_store (
    id          SERIAL PRIMARY KEY,
    question    TEXT NOT NULL,
    answer      TEXT,                       -- NULL when status = pending
    embedding   vector(1536),              -- embedding of the question for similarity match
    status      TEXT DEFAULT 'pending',    -- 'pending' | 'answered'
    created_at  TIMESTAMP DEFAULT NOW(),
    updated_at  TIMESTAMP DEFAULT NOW()
);

-- Index for fast Q&A similarity search
CREATE INDEX IF NOT EXISTS idx_qa_store_embedding
ON qa_store USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 50);
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Neon PostgreSQL connection string
# Get this from: Neon Console → Your Project → Connection String
DATABASE_URL=postgresql://user:password@ep-xxxx.us-east-2.aws.neon.tech/neondb?sslmode=require

# Admin password to access the Admin tab in Gradio
ADMIN_PASSWORD=your_secure_admin_password
```

**For Hugging Face Spaces**, add these as **Secrets** under Settings → Variables and Secrets (not in the repo).

---

## Python Dependencies (`requirements.txt`)

```
gradio>=4.0.0
openai>=1.0.0
psycopg2-binary>=2.9.0
pgvector>=0.2.0
pymupdf>=1.23.0          # PDF parsing (fitz)
python-docx>=1.0.0       # DOCX parsing
python-dotenv>=1.0.0
numpy>=1.24.0
tiktoken>=0.5.0          # Token counting for chunking
```

---

## Module Responsibilities

### `persona.py`
Defines the system prompt used for every chat call. **No personal details are hardcoded here** — all profile information is injected dynamically from RAG search results at query time. This means you only need to update your documents to keep the bot current.

```python
SYSTEM_PROMPT = """
You are an AI assistant representing a professional's portfolio.
You answer questions on behalf of the person whose documents have been uploaded.

## Your Role
- Answer questions about this person's skills, experience, projects, and background
- Speak in first person as if you are the person themselves
- Be professional, conversational, and concise

## How You Get Information
- All details about this person are provided to you in the [CONTEXT] block below
- The context is retrieved from their uploaded documents (resume, project docs, etc.)
- The context may also include previously answered Q&A pairs
- Do NOT make up or assume any facts not present in the context

## Answering Rules
- If the context contains a clear answer → respond confidently using it
- If the context is partially relevant → use what is available and be transparent
  about what you are unsure of
- If the context has no relevant information → say honestly:
  "I don't have details on that right now. You can reach out directly via the
  contact information in my profile."
- Never fabricate skills, projects, dates, or achievements
- If asked about availability or hiring → refer only to what the documents say;
  if not mentioned, say you are open to discussing opportunities

## Tone
- First person ("I", "my", "I have worked on...")
- Professional but approachable
- Concise — no unnecessary filler or repetition

## What You Help With
- Technical skills and tools
- Work experience and roles
- Projects and achievements
- Certifications and education
- How to get in touch

The context retrieved for this question is provided below. Use it as your
only source of facts about this person.
"""
```

> **Why this approach is better:** You never need to touch `persona.py` again. Just upload an updated resume or add a new project document and the bot will automatically reflect the changes on the next query.

---

### `db.py`
Handles all PostgreSQL interactions.

**Functions to implement:**

```python
def get_connection()
    # Returns a psycopg2 connection using DATABASE_URL from env

def store_chunk(source: str, chunk_index: int, content: str, embedding: list[float])
    # Inserts a document chunk with its embedding into document_chunks

def search_similar_chunks(query_embedding: list[float], top_k: int = 5) -> list[dict]
    # Cosine similarity search on document_chunks, returns top_k results
    # SQL: ORDER BY embedding <=> %s::vector LIMIT %s

def search_qa_store(query_embedding: list[float], threshold: float = 0.85) -> dict | None
    # Searches qa_store for answered questions similar to query
    # Only returns rows where status = 'answered' and similarity >= threshold
    # SQL: WHERE status = 'answered' ORDER BY embedding <=> %s::vector LIMIT 1

def log_unanswered_question(question: str, embedding: list[float])
    # Inserts question into qa_store with status = 'pending', answer = NULL

def get_unanswered_questions() -> list[dict]
    # Returns all rows from qa_store where status = 'pending'

def update_answer(qa_id: int, answer: str)
    # Updates qa_store: set answer = %s, status = 'answered', updated_at = NOW()
    # Also re-embed the question if needed

def get_all_qa_pairs() -> list[dict]
    # Returns all rows from qa_store ordered by created_at DESC

def delete_qa_pair(qa_id: int)
    # Deletes a row from qa_store by id

def chunk_already_exists(source: str, chunk_index: int) -> bool
    # Check if a chunk from this source file already exists (avoid re-ingestion)
```

---

### `rag.py`
Handles document ingestion and retrieval.

**Functions to implement:**

```python
def get_embedding(text: str) -> list[float]
    # Calls OpenAI text-embedding-3-small and returns the embedding vector

def parse_document(file_path: str) -> str
    # Dispatches to correct parser based on file extension
    # Supports: .pdf (PyMuPDF), .docx (python-docx), .txt (open)

def chunk_text(text: str, max_tokens: int = 400, overlap: int = 50) -> list[str]
    # Splits text into overlapping chunks using tiktoken
    # Use cl100k_base encoding

def ingest_documents(folder_path: str)
    # Iterates over all files in folder, parses, chunks, embeds, stores
    # Skips files already ingested (use chunk_already_exists check)

def ingest_single_file(file_path: str) -> str
    # Ingests a single uploaded file, returns status message

def retrieve_context(query: str, top_k: int = 5) -> str
    # Embeds query, searches document_chunks, returns formatted context string
```

---

### `chatbot.py`
Core query routing and response generation.

**Functions to implement:**

```python
def build_messages(user_query: str, history: list, rag_context: str, qa_answer: str | None) -> list[dict]
    # Constructs the messages array for OpenAI chat completion
    # System prompt is always the generic SYSTEM_PROMPT from persona.py
    # Personal details come ONLY from rag_context and qa_answer — never hardcoded
    # Priority: qa_answer (exact/similar match from Q&A store) > rag_context (doc chunks) > honest "I don't know"
    # Inject context into the user message as:
    #   "[CONTEXT]\n{rag_context or qa_answer}\n\n[QUESTION]\n{user_query}"
    # Includes last 6 turns from history for memory

def stream_response(user_query: str, history: list) -> Generator[str, None, None]
    # Main entry point called by Gradio
    # 1. Embed the query
    # 2. Search qa_store for similar answered question → if found, use it directly
    # 3. Search document_chunks for relevant context
    # 4. Call GPT-4o-mini with streaming=True
    # 5. Assess confidence from response (check for uncertainty phrases)
    # 6. If low confidence → log to qa_store as pending
    # 7. Yield tokens

UNCERTAINTY_PHRASES = [
    "i don't know", "i'm not sure", "i cannot find",
    "not mentioned", "no information", "i don't have details"
]
# If any phrase appears in the final response → log as unanswered
```

---

### `app.py`
Main Gradio application with three tabs.

**Tab 1 — Chat**
```
- gr.Chatbot for conversation display
- gr.Textbox for user input
- gr.Button "Send"
- gr.Button "Download Resume" (links to docs/Chandrakanth_K_Resume.pdf)
- gr.Button "View LinkedIn" (links to LinkedIn URL)
- gr.File for document upload (multiple=True)
- gr.Button "Upload & Ingest Documents"
- gr.Textbox (output) for ingestion status
```

**Tab 2 — Admin (Password Protected)**
```
- gr.Textbox for admin password entry
- gr.Button "Login"
- On successful login, reveal:
    - gr.Dropdown listing all pending questions (refresh on load)
    - gr.Textbox for typing the answer
    - gr.Button "Save Answer"
    - gr.Button "Refresh List"
    - gr.HTML showing full Q&A table (all pairs, all statuses)
    - gr.Number for Q&A ID to delete
    - gr.Button "Delete Entry"
```

**Tab 3 — About This App**
```
- Static markdown explaining how the chatbot works
- Tech stack used
- Note about RAG and Q&A learning loop
```

---

## RAG Query Flow (Detailed)

```
User asks: "What tools does Chandrakanth know?"

Step 1 — Embed query using text-embedding-3-small

Step 2 — Search qa_store
    → Look for answered questions with cosine similarity >= 0.85
    → If found: inject as authoritative answer, skip LLM retrieval

Step 3 — Search document_chunks
    → Top 5 chunks by cosine similarity
    → Format as context string

Step 4 — Build messages for GPT-4o-mini
    → system: SYSTEM_PROMPT
    → user: "Context:\n{rag_context}\n\nQuestion: {user_query}"
    → include last 6 turns of history

Step 5 — Stream response

Step 6 — Post-response confidence check
    → Scan response text for UNCERTAINTY_PHRASES
    → If uncertain: call log_unanswered_question(query, embedding)
    → Show message: "I've noted this question for Chandrakanth to answer."
```

---

## Hugging Face Spaces Setup

1. Create a new Space at https://huggingface.co/spaces
2. Select **Gradio** as the SDK
3. Choose **CPU Basic** (free tier)
4. Push your code via Git:
   ```bash
   git init
   git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/chandrakanth-portfolio
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```
5. Add Secrets in Space Settings:
   - `OPENAI_API_KEY`
   - `DATABASE_URL`
   - `ADMIN_PASSWORD`
6. Place your resume PDF in the `docs/` folder and commit it
7. The Space auto-builds and runs `app.py`

---

## Neon PostgreSQL Setup

1. Sign up at https://neon.tech (free tier, no credit card needed)
2. Create a new project → copy the **Connection String**
3. Open the **SQL Editor** in Neon console
4. Run all SQL statements from the **PostgreSQL Schema** section above
5. Paste the connection string as `DATABASE_URL` in your `.env` and HF Secrets

---

## Local Development

```bash
# Clone your repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/chandrakanth-portfolio
cd chandrakanth-portfolio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your actual keys

# Place your resume in docs/
mkdir docs
cp /path/to/Chandrakanth_K_Resume.pdf docs/

# Run the app
python app.py
# Opens at http://localhost:7860
```

---

## Important Notes for Claude Code

- **Never expose `OPENAI_API_KEY` or `DATABASE_URL` in the frontend** — all OpenAI and DB calls happen in Python backend only
- **Use `pgvector` Python package** alongside `psycopg2-binary` for vector type handling in queries — register the vector type with `register_vector(conn)` from `pgvector.psycopg2`
- **Streaming in Gradio** — use `gr.ChatInterface` with a generator function or wire manually with `.stream()` on the OpenAI client
- **Chunk overlap** — implement sliding window chunking (400 tokens, 50 token overlap) to avoid losing context at chunk boundaries
- **Re-ingestion guard** — always check `chunk_already_exists()` before ingesting to avoid duplicate vectors
- **Admin password** — compare against `os.getenv("ADMIN_PASSWORD")` server-side; never hardcode
- **Connection pooling** — for Neon serverless, open a new connection per request or use a simple pool; avoid persistent connections that timeout
- **SSL** — Neon requires `sslmode=require` in the connection string; ensure this is present
- **Error handling** — wrap all DB and OpenAI calls in try/except; surface friendly errors in the Gradio UI

---

## Example Questions the Bot Should Handle

| Question | Source |
|---|---|
| "What tools and technologies do you know?" | Resume RAG (doc chunks) |
| "Tell me about the Snowflake migration project" | Resume RAG (doc chunks) |
| "Are you open to new opportunities?" | Resume RAG (if mentioned in docs) |
| "What certifications do you have?" | Resume RAG (doc chunks) |
| "How can I contact you?" | Resume RAG (contact section of resume) |
| "What did you build at OneIntegral?" | Resume RAG (doc chunks) |
| "Do you know Kafka?" | RAG → if not in docs → logged as unanswered |
| "What is your notice period?" | Q&A Store (after you answer it once in admin tab) |

> **Key principle:** The bot never invents an answer. If it is not in the uploaded documents or the Q&A store, the question gets logged as pending for you to answer via the admin tab.

---

## File: `.env.example`

```env
OPENAI_API_KEY=sk-your-openai-key-here
DATABASE_URL=postgresql://user:password@ep-xxxx.region.aws.neon.tech/neondb?sslmode=require
ADMIN_PASSWORD=change-this-to-something-strong
```

---

*Built for Chandrakanth Karunakaran — Lead Data Engineer, Chennai, India*
*LinkedIn: https://www.linkedin.com/in/chandrakanthkarunakaran/*