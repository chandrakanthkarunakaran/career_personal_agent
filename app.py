import os
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from chatbot import stream_response
from rag import ingest_documents, ingest_single_file
from admin import (
    check_admin_password,
    load_pending_questions,
    save_answer,
    build_qa_table_html,
    delete_entry,
)

DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "docs")
RESUME_PATH = os.path.join(DOCS_FOLDER, "Chandrakanth_K_Resume.pdf")
LINKEDIN_URL = "https://www.linkedin.com/in/chandrakanthkarunakaran/"

# Pre-load documents at startup
if os.path.isdir(DOCS_FOLDER):
    try:
        ingest_documents(DOCS_FOLDER)
    except Exception as e:
        print(f"Startup ingestion error: {e}")


def chat(message: str, history: list):
    history = history or []
    partial = ""
    for partial in stream_response(message, history):
        yield partial


def upload_and_ingest(files) -> str:
    if not files:
        return "No files uploaded."
    statuses = []
    for file in files:
        path = file.name if hasattr(file, "name") else str(file)
        statuses.append(ingest_single_file(path))
    return "\n".join(statuses)


def admin_login(password: str):
    if check_admin_password(password):
        pending = load_pending_questions()
        dropdown_choices = pending if pending else ["No pending questions"]
        table_html = build_qa_table_html()
        return (
            gr.update(visible=True),   # admin_panel
            gr.update(choices=dropdown_choices, value=None),  # pending_dropdown
            table_html,                # qa_table
            "Login successful.",       # login_status
        )
    return (
        gr.update(visible=False),
        gr.update(choices=[]),
        "",
        "Incorrect password.",
    )


def refresh_pending():
    pending = load_pending_questions()
    choices = pending if pending else ["No pending questions"]
    return gr.update(choices=choices, value=None), build_qa_table_html()


def handle_save_answer(selected: str, answer: str):
    msg = save_answer(selected, answer)
    pending = load_pending_questions()
    choices = pending if pending else ["No pending questions"]
    return msg, gr.update(choices=choices, value=None), build_qa_table_html()


def handle_delete(qa_id):
    msg = delete_entry(qa_id)
    return msg, build_qa_table_html()


ABOUT_TEXT = """
## About This App

This is an AI-powered portfolio chatbot for **Chandrakanth Karunakaran**, Lead Data Engineer.

### How It Works

1. **RAG (Retrieval-Augmented Generation)** — When you ask a question, the bot searches
   uploaded documents (resume, project docs) using vector similarity and injects relevant
   excerpts as context for GPT-4o-mini.

2. **Q&A Store** — Human-answered Q&A pairs are stored in PostgreSQL. If your question
   closely matches a previously answered one (≥85% similarity), the stored answer is used
   directly without hitting the LLM.

3. **Human-in-the-Loop Learning** — Questions the bot can't confidently answer are logged
   as "pending". Once answered via the Admin tab, they feed back into the Q&A store and
   are reused for similar future questions.

### Tech Stack

| Layer | Technology |
|---|---|
| UI | Gradio 4.x |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | text-embedding-3-small |
| Vector Store | pgvector on Neon PostgreSQL |
| Document Parsing | PyMuPDF, python-docx |
| Hosting | Hugging Face Spaces |

### Key Principle
The bot never invents answers. Everything it says comes from uploaded documents or
manually approved Q&A pairs.
"""

with gr.Blocks(title="Chandrakanth K — AI Portfolio Chatbot") as demo:
    gr.Markdown("# Chandrakanth K — AI Portfolio Chatbot")
    gr.Markdown("Ask me anything about my skills, experience, and projects.")

    with gr.Tabs():
        # --- Tab 1: Chat ---
        with gr.Tab("Chat"):
            chatbot_ui = gr.Chatbot(height=480)
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask something about Chandrakanth...",
                    show_label=False,
                    scale=8,
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")

            with gr.Row():
                if os.path.isfile(RESUME_PATH):
                    resume_btn = gr.DownloadButton(
                        label="Download Resume",
                        value=RESUME_PATH,
                        variant="secondary",
                    )
                linkedin_btn = gr.Button("View LinkedIn", variant="secondary")

            def _extract_text(content) -> str:
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            parts.append(item["text"])
                        elif isinstance(item, str):
                            parts.append(item)
                    return " ".join(parts)
                return str(content)

            def _to_plain_history(history: list) -> list:
                plain = []
                for msg in history:
                    plain.append({"role": msg["role"], "content": _extract_text(msg["content"])})
                return plain

            def user_submit(message, history):
                history = history or []
                history.append({"role": "user", "content": message})
                return "", history

            def bot_respond(history):
                if not history:
                    return history
                last_user = _extract_text(history[-1]["content"])
                plain_history = _to_plain_history(history[:-1])
                history.append({"role": "assistant", "content": ""})
                for partial in stream_response(last_user, plain_history):
                    history[-1]["content"] = partial
                    yield history

            msg_box.submit(user_submit, [msg_box, chatbot_ui], [msg_box, chatbot_ui]).then(
                bot_respond, [chatbot_ui], [chatbot_ui]
            )
            send_btn.click(user_submit, [msg_box, chatbot_ui], [msg_box, chatbot_ui]).then(
                bot_respond, [chatbot_ui], [chatbot_ui]
            )
            linkedin_btn.click(
                fn=lambda: None,
                js=f"() => {{ window.open('{LINKEDIN_URL}', '_blank'); }}",
            )

        # --- Tab 2: Admin ---
        with gr.Tab("Admin"):
            gr.Markdown("## Admin Panel")
            gr.Markdown("Enter the admin password to manage unanswered questions.")

            with gr.Row():
                password_box = gr.Textbox(
                    label="Admin Password",
                    type="password",
                    placeholder="Enter password...",
                    scale=4,
                )
                login_btn = gr.Button("Login", variant="primary", scale=1)
            login_status = gr.Textbox(label="Status", interactive=False)

            with gr.Column(visible=False) as admin_panel:
                gr.Markdown("### Pending Questions")
                with gr.Row():
                    pending_dropdown = gr.Dropdown(
                        label="Select a pending question",
                        choices=[],
                        scale=6,
                    )
                    refresh_btn = gr.Button("Refresh List", scale=1)

                answer_box = gr.Textbox(
                    label="Your Answer",
                    placeholder="Type the answer here...",
                    lines=4,
                )
                save_btn = gr.Button("Save Answer", variant="primary")
                save_status = gr.Textbox(label="Save Status", interactive=False)

                gr.Markdown("---")
                gr.Markdown("### All Q&A Pairs")
                qa_table = gr.HTML()

                gr.Markdown("### Delete Entry")
                with gr.Row():
                    delete_id = gr.Number(label="Q&A ID to delete", precision=0, scale=3)
                    delete_btn = gr.Button("Delete Entry", variant="stop", scale=1)
                delete_status = gr.Textbox(label="Delete Status", interactive=False)

                gr.Markdown("---")
                gr.Markdown("### Upload Documents")
                file_upload = gr.File(
                    label="Upload PDFs, DOCXs, or TXTs",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".txt"],
                )
                upload_btn = gr.Button("Upload & Ingest Documents", variant="secondary")
                ingest_status = gr.Textbox(label="Ingestion Status", interactive=False)

            login_btn.click(
                admin_login,
                inputs=[password_box],
                outputs=[admin_panel, pending_dropdown, qa_table, login_status],
            )
            refresh_btn.click(
                refresh_pending,
                outputs=[pending_dropdown, qa_table],
            )
            save_btn.click(
                handle_save_answer,
                inputs=[pending_dropdown, answer_box],
                outputs=[save_status, pending_dropdown, qa_table],
            )
            delete_btn.click(
                handle_delete,
                inputs=[delete_id],
                outputs=[delete_status, qa_table],
            )
            upload_btn.click(upload_and_ingest, inputs=[file_upload], outputs=[ingest_status])

        # --- Tab 3: About ---
        with gr.Tab("About This App"):
            gr.Markdown(ABOUT_TEXT)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), server_name="0.0.0.0")
