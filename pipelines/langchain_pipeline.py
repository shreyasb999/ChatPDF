# pipelines/langchain_pipeline.py

import os
import uuid
from typing import List, Tuple
from base64 import b64decode
from together import Together
from config.settings import LANGCHAIN_API_KEY, TOGETHER_API_KEY
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever


# ────────────────────────────────────────────────────────────────────────────────
# 1) Helper: call Together AI REST endpoint directly
# ────────────────────────────────────────────────────────────────────────────────

if not TOGETHER_API_KEY:
    raise RuntimeError("Please set TOGETHER_API_KEY in your environment")

# Correct OpenAI‐compatible endpoint
TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"

def complete_with_together(
    prompt: str,
    model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    max_tokens: int = 256,
    temperature: float = 0.5,
    stop_sequences: List[str] = None,
) -> str:
    """
    Make a chat‐completion request to Together’s /v1/chat/completions.
    Returns the assistant’s reply as text.
    """
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stop_sequences:
        body["stop"] = stop_sequences

    resp = requests.post(TOGETHER_URL, headers=headers, json=body)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ────────────────────────────────────────────────────────────────────────────────
# 2) Summarization & “Image Description” via Together REST
# ────────────────────────────────────────────────────────────────────────────────

def summarize_element(element: str) -> str:
    """
    Summarize a single text/table chunk via Together AI.
    """
    prompt = f"""
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text—no extra commentary.
Table or text chunk: {element}
"""
    return complete_with_together(prompt=prompt, max_tokens=256, temperature=0.5)


def describe_image_placeholder() -> str:
    """
    “Describe” a generic transformer diagram via Together AI.
    """
    prompt = """
Describe in detail what a transformer architecture diagram typically looks like
in an academic research paper—stacked transformer blocks, multi-head attention,
feed-forward layers, etc. You do not have raw pixel data, so rely on common conventions.
"""
    return complete_with_together(prompt=prompt, max_tokens=256, temperature=0.5)


def summarize_chunks(
    texts: List[str],
    tables: List[str],
    images_b64: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Given:
      - texts:      list of text chunks
      - tables:     list of HTML‐serialized tables
      - images_b64: list of base64‐encoded image strings

    Returns three lists of summaries: (text_summaries, table_summaries, image_summaries).
    """
    text_summaries = [summarize_element(t) for t in texts] if texts else []
    table_summaries = [summarize_element(tbl_html) for tbl_html in tables] if tables else []
    image_summaries = [describe_image_placeholder() for _ in images_b64] if images_b64 else []
    return text_summaries, table_summaries, image_summaries


# ────────────────────────────────────────────────────────────────────────────────
# 3) Build Retriever (Vector Index) with Chroma + InMemoryStore
# ────────────────────────────────────────────────────────────────────────────────

def build_retriever(
    text_summaries: List[str],
    table_summaries: List[str],
    image_summaries: List[str],
    original_texts: List[str],
    original_tables: List[str],
    original_images: List[str],
    collection_name: str = "multi_modal_rag",
) -> MultiVectorRetriever:
    """
    Index non‐empty summaries in Chroma (with HuggingFaceEmbeddings),
    and store mapping back to original payloads in InMemoryStore.

    Any empty summary list is skipped to avoid upsert errors.
    """
    chroma_vs = Chroma(
        collection_name=collection_name,
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    )
    docstore = InMemoryStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(vectorstore=chroma_vs, docstore=docstore, id_key=id_key)

    def _add_batch(summaries: List[str], originals: List[str]):
        # Skip empty lists
        if not summaries or not originals:
            return

        ids = [str(uuid.uuid4()) for _ in summaries]
        docs = [
            Document(page_content=summaries[i], metadata={id_key: ids[i]})
            for i in range(len(summaries))
        ]
        retriever.vectorstore.add_documents(docs)
        retriever.docstore.mset(list(zip(ids, originals)))

    # Only index if summaries are non‐empty
    _add_batch(text_summaries, original_texts)
    _add_batch(table_summaries, original_tables)
    _add_batch(image_summaries, original_images)

    return retriever


# ────────────────────────────────────────────────────────────────────────────────
# 4) Build & Ask RAG Chain (via Together REST)
# ────────────────────────────────────────────────────────────────────────────────

def build_rag_chain(retriever: MultiVectorRetriever):
    """
    Return two functions for RAG:
      - retrieve_fn(query): returns (list_of_texts, list_of_image_placeholders)
      - answer_fn(question, texts, images): returns the final answer string
    """
    def retrieve_fn(query: str) -> Tuple[List[str], List[str]]:
        docs = retriever.get_relevant_documents(query)
        texts, images = [], []
        for doc in docs:
            payload = retriever.docstore.get(doc.metadata["doc_id"])
            try:
                b64decode(payload)
                images.append(payload)
            except Exception:
                texts.append(payload)
        return texts, images

    def answer_fn(question: str, context_texts: List[str], context_images: List[str]) -> str:
        prompt = f"""
You are a helpful assistant. Answer the user’s question based on the provided context.

Context Texts:
{context_texts}

Context Images (base64 placeholders):
{context_images}

Question: {question}

Provide a concise answer, referencing context as needed.
"""
        return complete_with_together(prompt=prompt, max_tokens=256, temperature=0.3)

    return retrieve_fn, answer_fn


def ask_question(
    question: str, retrieve_fn, answer_fn
) -> Tuple[str, List[str], List[str]]:
    """
    Run retrieval, then answer via Together.
    Returns (answer_text, context_texts, context_images).
    """
    context_texts, context_images = retrieve_fn(question)
    answer = answer_fn(question, context_texts, context_images)
    return answer, context_texts, context_images


# ────────────────────────────────────────────────────────────────────────────────
# 5) High-Level Helper: build_pipeline()
# ────────────────────────────────────────────────────────────────────────────────

def build_pipeline(
    text_chunks: List[str], table_htmls: List[str], image_b64s: List[str]
):
    """
    Summarize each modality via Together, index non-empty summaries
    in Chroma, and return (retrieve_fn, answer_fn).
    """
    txt_summaries, tbl_summaries, img_summaries = summarize_chunks(
        text_chunks, table_htmls, image_b64s
    )
    retr = build_retriever(
        txt_summaries, tbl_summaries, img_summaries,
        text_chunks, table_htmls, image_b64s
    )
    return build_rag_chain(retr)