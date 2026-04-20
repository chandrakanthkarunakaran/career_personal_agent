---
title: Chandrakanth K Portfolio Chatbot
emoji: 💼
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.12.0"
app_file: app.py
pinned: true
---

# Chandrakanth K — AI Portfolio Chatbot

An agentic AI chatbot that answers questions about Chandrakanth Karunakaran's skills,
experience, and projects using RAG over his resume and a human-in-the-loop Q&A store.

## Features
- GPT-4o-mini with agentic tool-use loop (Q&A store → resume RAG → stream answer)
- pgvector on Neon PostgreSQL for semantic search
- Admin panel (password-protected) to answer logged questions
- Resume download button

## Secrets required (set in Space Settings → Variables and Secrets)
- `OPENAI_API_KEY`
- `DATABASE_URL`
- `ADMIN_PASSWORD`
