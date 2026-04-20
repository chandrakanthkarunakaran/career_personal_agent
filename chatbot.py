import json
from typing import Generator
from openai import OpenAI
from persona import SYSTEM_PROMPT
from rag import get_embedding, retrieve_context
from db import search_qa_store as db_search_qa, log_unanswered_question as db_log_unanswered

client = OpenAI()

MAX_TOOL_ITERATIONS = 5

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_qa_store",
            "description": (
                "Search previously human-answered Q&A pairs for a matching answer. "
                "Always try this FIRST — a human-approved answer is more authoritative than document retrieval."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The user's question to search for a prior answer"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_resume_docs",
            "description": (
                "Search Chandrakanth's resume and uploaded project documents using semantic similarity. "
                "Use this for questions about skills, experience, projects, education, certifications, "
                "tools, contact info, or any factual detail about his background."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A descriptive search query to find relevant document chunks"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_unanswered_question",
            "description": (
                "Log a question as pending for human review when neither the Q&A store nor the documents "
                "contain a confident answer. Call this before giving an 'I don't have details' response."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The exact user question to log for Chandrakanth to answer"}
                },
                "required": ["question"],
            },
        },
    },
]


# --- Tool implementations ---

def _search_qa_store(query: str) -> str:
    try:
        embedding = get_embedding(query)
        result = db_search_qa(embedding, threshold=0.85)
        if result:
            return f"[Authoritative Answer — similarity {result['similarity']:.2f}]\n{result['answer']}"
        return "No matching answered question found in Q&A store."
    except Exception as e:
        return f"Q&A store search error: {e}"


def _search_resume_docs(query: str) -> str:
    try:
        context = retrieve_context(query, top_k=5)
        return context if context else "No relevant document chunks found."
    except Exception as e:
        return f"Document search error: {e}"


def _log_unanswered_question(question: str) -> str:
    try:
        embedding = get_embedding(question)
        db_log_unanswered(question, embedding)
        return "Question logged for Chandrakanth to review and answer."
    except Exception as e:
        return f"Failed to log question: {e}"


TOOL_DISPATCH = {
    "search_qa_store": _search_qa_store,
    "search_resume_docs": _search_resume_docs,
    "log_unanswered_question": _log_unanswered_question,
}


def _execute_tool(name: str, arguments: dict) -> str:
    fn = TOOL_DISPATCH.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    try:
        return fn(**arguments)
    except Exception as e:
        return f"Tool execution error: {e}"


def _msg_to_dict(msg) -> dict:
    d = {"role": msg.role, "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return d


def _build_messages(user_query: str, history: list) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history[-6:]:
        if isinstance(turn, dict):
            messages.append({"role": turn["role"], "content": turn["content"]})
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            u, a = turn
            if u:
                messages.append({"role": "user", "content": u})
            if a:
                messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_query})
    return messages


UNCERTAINTY_PHRASES = [
    "i don't know",
    "i'm not sure",
    "i cannot find",
    "not mentioned",
    "no information",
    "i don't have details",
    "i don't have specific details",
    "i don't have information",
    "reach out directly via the contact",
    "contact information in my profile",
]


def stream_response(user_query: str, history: list) -> Generator[str, None, None]:
    messages = _build_messages(user_query, history)
    logged_question = False

    # --- Agentic tool-call loop ---
    # Iteration 0: always search Q&A store first (human answers are authoritative).
    # Iteration 1: always search docs (ensures RAG context even when Q&A misses).
    # Iteration 2+: auto — model decides if more tools are needed.
    qa_had_no_match = False
    for iteration in range(MAX_TOOL_ITERATIONS):
        if iteration == 0:
            tool_choice = {"type": "function", "function": {"name": "search_qa_store"}}
        elif iteration == 1 and qa_had_no_match:
            tool_choice = {"type": "function", "function": {"name": "search_resume_docs"}}
        else:
            tool_choice = "auto"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice=tool_choice,
            temperature=0.4,
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if not msg.tool_calls:
            # No tools requested — model is ready to give the final answer.
            break

        # Append the assistant's tool-call turn
        messages.append(_msg_to_dict(msg))

        # Execute every requested tool and append results
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            result = _execute_tool(name, args)

            if name == "log_unanswered_question":
                logged_question = True
            if name == "search_qa_store" and "No matching" in result:
                qa_had_no_match = True

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    # --- Stream the final answer ---
    # messages now ends with tool results; ask the model to compose the reply.
    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="none",   # force text-only reply using all collected context
            stream=True,
            temperature=0.4,
        )
    except Exception as e:
        yield f"Error generating response: {e}"
        return

    full_response = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full_response += delta
            yield full_response

    # Fallback: if the model expressed uncertainty but didn't call log_unanswered_question,
    # log it now so no unanswered question slips through.
    if not logged_question and any(p in full_response.lower() for p in UNCERTAINTY_PHRASES):
        try:
            embedding = get_embedding(user_query)
            db_log_unanswered(user_query, embedding)
            logged_question = True
        except Exception:
            pass

    if logged_question:
        yield full_response + "\n\n_I've noted this question for Chandrakanth to answer._"
