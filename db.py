import os
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    return conn


def store_chunk(source: str, chunk_index: int, content: str, embedding: list[float]):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_chunks (source, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (source, chunk_index, content, embedding),
            )
        conn.commit()
    finally:
        conn.close()


def search_similar_chunks(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, source, chunk_index, content,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM document_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def search_qa_store(query_embedding: list[float], threshold: float = 0.85) -> dict | None:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, question, answer,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM qa_store
                WHERE status = 'answered'
                ORDER BY embedding <=> %s::vector
                LIMIT 1
                """,
                (query_embedding, query_embedding),
            )
            row = cur.fetchone()
            if row and row["similarity"] >= threshold:
                return dict(row)
            return None
    finally:
        conn.close()


def log_unanswered_question(question: str, embedding: list[float]):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO qa_store (question, answer, embedding, status)
                VALUES (%s, NULL, %s, 'pending')
                """,
                (question, embedding),
            )
        conn.commit()
    finally:
        conn.close()


def get_unanswered_questions() -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, question, created_at FROM qa_store WHERE status = 'pending' ORDER BY created_at DESC"
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def update_answer(qa_id: int, answer: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE qa_store
                SET answer = %s, status = 'answered', updated_at = NOW()
                WHERE id = %s
                """,
                (answer, qa_id),
            )
        conn.commit()
    finally:
        conn.close()


def get_all_qa_pairs() -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, question, answer, status, created_at, updated_at FROM qa_store ORDER BY created_at DESC"
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def delete_qa_pair(qa_id: int):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM qa_store WHERE id = %s", (qa_id,))
        conn.commit()
    finally:
        conn.close()


def chunk_already_exists(source: str, chunk_index: int) -> bool:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM document_chunks WHERE source = %s AND chunk_index = %s LIMIT 1",
                (source, chunk_index),
            )
            return cur.fetchone() is not None
    finally:
        conn.close()
