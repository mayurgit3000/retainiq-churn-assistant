"""
rag_engine.py — RetainIQ RAG Query Engine
Retrieves relevant chunks from FAISS, calls OpenRouter LLM, returns answer + citations.
"""

import requests
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

VECTORSTORE_DIR = Path("vectorstore")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "anthropic/claude-3-haiku"
TOP_K = 4

SOURCE_LABELS = {
    "churnsense_eda_findings.txt": "ChurnSense — EDA & Cohort Analysis",
    "churnsense_shap_model_report.txt": "ChurnSense — SHAP & Model Performance",
    "retainiq_retention_playbook.txt": "RetainIQ — Retention Playbook",
}

SYSTEM_PROMPT = """You are the RetainIQ Assistant — an AI-powered intelligence layer built on top of ChurnSense, a customer churn prediction model.

You help retention agents, analysts, and managers understand:
- Why customers churn (based on ChurnSense EDA and SHAP findings)
- Which customers are highest risk and why
- What retention actions to take for specific customer profiles
- How the ChurnSense model works and what its outputs mean
- Business impact, ROI, and strategic recommendations

Rules:
1. Answer ONLY using information from the provided context documents.
2. If the context does not contain enough information, respond: "I don't have enough information in the RetainIQ knowledge base to answer that. Please refer to the ChurnSense technical documentation or contact the data science team."
3. Be concise, direct, and actionable. Use bullet points for lists of actions or findings.
4. Always connect insights to business value where possible.
5. Never fabricate statistics, SHAP values, or model metrics not present in the context."""

def load_vectorstore():
    # allow_dangerous_deserialization is safe here because the index is built
    # locally by ingest.py. Never load a FAISS index from an untrusted source.
    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError("Vector store not found. Run: python ingest.py")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(
        str(VECTORSTORE_DIR), embeddings, allow_dangerous_deserialization=True
    )

def retrieve(vectorstore, query):
    return vectorstore.similarity_search_with_score(query, k=TOP_K)

def build_context(results):
    parts = []
    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get("source", "Unknown")
        label = SOURCE_LABELS.get(source, source)
        parts.append(f"[Source {i+1}: {label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)

def call_llm(query, context, api_key):
    user_message = f"""Use the following context documents to answer the question.

CONTEXT:
{context}

QUESTION: {query}

Answer based only on the context above:"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "RetainIQ Assistant",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.2,
        "max_tokens": 800,
    }
    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text}")
    return response.json()["choices"][0]["message"]["content"]

def is_out_of_scope(results, threshold=1.2):
    if not results:
        return True
    return results[0][1] > threshold

def answer_query(vectorstore, query, api_key):
    results = retrieve(vectorstore, query)
    if is_out_of_scope(results):
        return {
            "answer": "I don't have enough information in the RetainIQ knowledge base to answer that. Please refer to the ChurnSense technical documentation or contact the data science team.",
            "sources": [],
            "out_of_scope": True,
        }
    context = build_context(results)
    answer = call_llm(query, context, api_key)
    seen = set()
    sources = []
    for doc, score in results:
        source = doc.metadata.get("source", "Unknown")
        label = SOURCE_LABELS.get(source, source)
        if label not in seen:
            seen.add(label)
            excerpt = doc.page_content[:220].strip().replace("\n", " ") + "..."
            sources.append({"label": label, "excerpt": excerpt})
    return {"answer": answer, "sources": sources, "out_of_scope": False}
