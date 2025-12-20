import os
import pickle
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import gradio as gr

# -------------------------------------------------------------------------
# 1. Load Environment + Validate Key
# -------------------------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ Missing GROQ_API_KEY in environment. Add it to .env")

client = Groq(api_key=GROQ_API_KEY)

# -------------------------------------------------------------------------
# 2. Load Preloaded Embeddings + Chunks
# -------------------------------------------------------------------------
EMB_FILE = "embeddings.pkl"
TXT_FILE = "chunks.pkl"

if not os.path.exists(EMB_FILE) or not os.path.exists(TXT_FILE):
    raise FileNotFoundError("❌ Missing embeddings/chunks. Run chunk + embed scripts first.")

with open(EMB_FILE, "rb") as f:
    embeddings = np.array(pickle.load(f))

with open(TXT_FILE, "rb") as f:
    chunks = pickle.load(f)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------------------------------------------------
# 3. Utility: LLM Call
# -------------------------------------------------------------------------
def ask_llm(prompt):
    """Call Groq LLM safely and return a string."""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ LLM Error: {str(e)}"

# -------------------------------------------------------------------------
# 4. Retrieve Top-k Chunks
# -------------------------------------------------------------------------
def retrieve(query, k=3):
    query_emb = embed_model.encode([query])[0]
    scores = np.dot(embeddings, query_emb)
    top_idx = scores.argsort()[-k:][::-1]
    # Filter: only keep chunks with meaningful similarity
    retrieved_chunks = [chunks[i] for i in top_idx if scores[i] > 0.1]
    return retrieved_chunks, top_idx

# -------------------------------------------------------------------------
# 5. Ask Question
# -------------------------------------------------------------------------
def answer_query(history, query):
    if not query.strip():
        return history + [{"role": "assistant", "content": "⚠️ Please enter a question."}], ""

    retrieved_texts, top_idx = retrieve(query)

    if not retrieved_texts:
        answer = "⚠️ I don't have content from the documents to answer that."
    else:
        context = "\n\n".join(retrieved_texts)
        prompt = f"""
You are an expert assistant. Use the DOCUMENT CONTEXT ONLY.
Answer based ONLY on the provided context.
If the context does not contain relevant information, respond with:
"I don't have content from the documents to answer that."

### CONTEXT
{context}

### QUESTION
{query}

### ANSWER
"""
        answer = ask_llm(prompt)

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
    return history, ""

# -------------------------------------------------------------------------
# 6. Summarize Topic (based on query & retrieved chunks)
# -------------------------------------------------------------------------
def summarize_topic(history, query):
    retrieved_texts, top_idx = retrieve(query)
    if not retrieved_texts:
        summary = "⚠️ I don't have content from the documents to summarize that topic."
    else:
        context = "\n\n".join(retrieved_texts)
        prompt = f"""
Summarize the following information in 3–5 bullet points.
Focus on key points relevant to the topic.
If the context does not contain relevant information, respond with:
"I don't have content from the documents to summarize that topic."

TEXT:
{context}
"""
        summary = ask_llm(prompt)

    history.append({"role": "user", "content": f"Summarize topic: {query}"})
    history.append({"role": "assistant", "content": summary})
    return history, ""

# -------------------------------------------------------------------------
# 7. Summarize Chat History
# -------------------------------------------------------------------------
def summarize_chat_history(history):
    if not history:
        return history, "⚠️ Chat history is empty."

    # Take last 40 entries or all
    chat_text = "\n".join([f"{h['role']}: {h['content']}" for h in history[-40:]])
    prompt = f"""
Summarize the following chat conversation in clear bullet points.
Focus on key topics, questions, and answers.
If the conversation includes questions without answers or out-of-document questions, note them as such.

CHAT:
{chat_text}
"""
    summary = ask_llm(prompt)

    history.append({"role": "assistant", "content": f"**Chat Summary:**\n{summary}"})
    return history, ""

# -------------------------------------------------------------------------
# 8. Clear Chat
# -------------------------------------------------------------------------
def clear_history():
    return [], ""

# -------------------------------------------------------------------------
# 9. Gradio Interface
# -------------------------------------------------------------------------
with gr.Blocks(title="RAG Document Chatbot") as app:
    gr.Markdown("<h2>📘 RAG Document Chatbot</h2>"
                "Ask questions, summarize topics, or summarize the chat history based on preloaded company PDFs.")

    chatbot = gr.Chatbot(label="Chatbot", height=400)
    query = gr.Textbox(label="Enter question or topic", placeholder="Type here...")

    with gr.Row():
        ask_btn = gr.Button("Ask Question")
        sum_btn = gr.Button("Summarize Topic")
        chat_sum_btn = gr.Button("Summarize Chat History")
        clr_btn = gr.Button("Clear Chat", variant="stop")

    # Button bindings
    ask_btn.click(answer_query, [chatbot, query], [chatbot, query])
    sum_btn.click(summarize_topic, [chatbot, query], [chatbot, query])
    chat_sum_btn.click(summarize_chat_history, [chatbot], [chatbot, query])
    clr_btn.click(clear_history, None, [chatbot, query])

# -------------------------------------------------------------------------
# 10. Launch App
# -------------------------------------------------------------------------
app.launch(server_port=7860, server_name="127.0.0.1")
