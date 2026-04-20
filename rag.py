import os
import fitz  # PyMuPDF
import tiktoken
from docx import Document as DocxDocument
from openai import OpenAI
from db import store_chunk, chunk_already_exists, search_similar_chunks

client = OpenAI()
_encoding = tiktoken.get_encoding("cl100k_base")


def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def parse_document(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        doc = fitz.open(file_path)
        return "\n".join(page.get_text() for page in doc)
    elif ext == ".docx":
        doc = DocxDocument(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, max_tokens: int = 400, overlap: int = 50) -> list[str]:
    tokens = _encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(_encoding.decode(chunk_tokens))
        if end == len(tokens):
            break
        start += max_tokens - overlap
    return chunks


def ingest_documents(folder_path: str):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in (".pdf", ".docx", ".txt"):
            continue
        try:
            text = parse_document(file_path)
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                if chunk_already_exists(filename, i):
                    continue
                embedding = get_embedding(chunk)
                store_chunk(filename, i, chunk, embedding)
            print(f"Ingested: {filename} ({len(chunks)} chunks)")
        except Exception as e:
            print(f"Failed to ingest {filename}: {e}")


def ingest_single_file(file_path: str) -> str:
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".pdf", ".docx", ".txt"):
        return f"Unsupported file type: {ext}"
    try:
        text = parse_document(file_path)
        chunks = chunk_text(text)
        ingested = 0
        for i, chunk in enumerate(chunks):
            if chunk_already_exists(filename, i):
                continue
            embedding = get_embedding(chunk)
            store_chunk(filename, i, chunk, embedding)
            ingested += 1
        if ingested == 0:
            return f"{filename}: already fully ingested (no new chunks)."
        return f"{filename}: ingested {ingested} new chunks."
    except Exception as e:
        return f"Error ingesting {filename}: {e}"


def retrieve_context(query: str, top_k: int = 5) -> str:
    embedding = get_embedding(query)
    results = search_similar_chunks(embedding, top_k=top_k)
    if not results:
        return ""
    parts = []
    for r in results:
        parts.append(f"[Source: {r['source']}]\n{r['content']}")
    return "\n\n---\n\n".join(parts)
