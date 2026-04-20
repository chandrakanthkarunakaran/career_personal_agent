import os
from db import get_unanswered_questions, update_answer, get_all_qa_pairs, delete_qa_pair


def check_admin_password(password: str) -> bool:
    expected = os.getenv("ADMIN_PASSWORD", "")
    return password == expected and expected != ""


def load_pending_questions() -> list[str]:
    try:
        rows = get_unanswered_questions()
        return [f"[{r['id']}] {r['question']}" for r in rows]
    except Exception as e:
        return [f"Error loading questions: {e}"]


def save_answer(selected: str, answer: str) -> str:
    if not selected or not answer.strip():
        return "Please select a question and enter an answer."
    try:
        qa_id = int(selected.split("]")[0].replace("[", "").strip())
        update_answer(qa_id, answer.strip())
        return f"Answer saved for question ID {qa_id}."
    except Exception as e:
        return f"Error saving answer: {e}"


def build_qa_table_html() -> str:
    try:
        rows = get_all_qa_pairs()
    except Exception as e:
        return f"<p>Error loading Q&A pairs: {e}</p>"

    if not rows:
        return "<p>No Q&A pairs found.</p>"

    html = """
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <thead>
        <tr style="background:#f0f0f0;">
          <th style="border:1px solid #ccc;padding:6px;">ID</th>
          <th style="border:1px solid #ccc;padding:6px;">Question</th>
          <th style="border:1px solid #ccc;padding:6px;">Answer</th>
          <th style="border:1px solid #ccc;padding:6px;">Status</th>
          <th style="border:1px solid #ccc;padding:6px;">Created</th>
        </tr>
      </thead>
      <tbody>
    """
    for r in rows:
        status_color = "#28a745" if r["status"] == "answered" else "#dc3545"
        answer_text = r["answer"] or "—"
        created = str(r["created_at"])[:19] if r["created_at"] else "—"
        html += f"""
        <tr>
          <td style="border:1px solid #ccc;padding:6px;text-align:center;">{r['id']}</td>
          <td style="border:1px solid #ccc;padding:6px;">{r['question']}</td>
          <td style="border:1px solid #ccc;padding:6px;">{answer_text}</td>
          <td style="border:1px solid #ccc;padding:6px;color:{status_color};font-weight:bold;">{r['status']}</td>
          <td style="border:1px solid #ccc;padding:6px;">{created}</td>
        </tr>
        """
    html += "</tbody></table>"
    return html


def delete_entry(qa_id: int) -> str:
    if not qa_id:
        return "Please enter a valid Q&A ID."
    try:
        delete_qa_pair(int(qa_id))
        return f"Deleted Q&A entry with ID {int(qa_id)}."
    except Exception as e:
        return f"Error deleting entry: {e}"
