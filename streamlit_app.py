# streamlit_app.py

import os
import streamlit as st
import pandas as pd

from config.settings import PDF_FILE
from processors.pdf_processor import load_and_partition_pdf, separate_chunks
from processors.image_utils import extract_base64_images, b64_to_pil
from utils.helpers import get_text_from_block

from pipelines.langchain_pipeline import build_pipeline, ask_question

st.set_page_config(page_title="PDF Explorer + RAG", layout="wide")

# ----------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_chunks(path: str):
    chunks = load_and_partition_pdf(path)
    texts, tables, images = separate_chunks(chunks)
    return texts, tables, images

@st.cache_data(show_spinner=False)
def init_rag_pipeline(_text_blocks, _table_blocks, _image_blocks):
    texts = [get_text_from_block(b) for b in _text_blocks]

    # Convert each table chunk to HTML
    table_htmls = []
    for tb in _table_blocks:
        try:
            df = tb.to_pandas()
        except Exception:
            df = pd.DataFrame(tb.rows or [])
        table_htmls.append(df.to_html(index=False, escape=False))

    images_b64 = extract_base64_images(_image_blocks)

    retrieve_fn, answer_fn = build_pipeline(texts, table_htmls, images_b64)
    return retrieve_fn, answer_fn

# ----------------------------------------------------------------
st.title("📄 PDF Explorer + RAG")

st.sidebar.header("Settings")
uploaded = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
if uploaded:
    tmp_path = "tmp.pdf"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())
    pdf_path = tmp_path
else:
    pdf_path = PDF_FILE

st.sidebar.markdown(f"**Using:** `{os.path.basename(pdf_path)}`")

with st.spinner("🔍 Partitioning PDF..."):
    text_blocks, table_blocks, image_blocks = load_chunks(pdf_path)

# Text Search
st.sidebar.subheader("🔎 Search Text")
query = st.sidebar.text_input("Search term:")
if query:
    hits = [
        get_text_from_block(b)
        for b in text_blocks
        if query.lower() in get_text_from_block(b).lower()
    ]
    st.subheader(f"Text Search Results ({len(hits)})")
    for i, h in enumerate(hits, 1):
        st.markdown(f"**{i}.** {h}")

# Browse Tables
st.sidebar.subheader("📊 Browse Tables")
if table_blocks:
    idx = st.sidebar.number_input("Table #", 1, len(table_blocks), 1)
    tb = table_blocks[idx - 1]
    try:
        df = tb.to_pandas()
    except Exception:
        df = pd.DataFrame(tb.rows or [])
    st.subheader(f"Table {idx}")
    st.dataframe(df)
else:
    st.sidebar.write("_No tables found._")

# Browse Images
st.sidebar.subheader("🖼️ Browse Images")
if image_blocks:
    idx = st.sidebar.number_input("Image #", 1, len(image_blocks), 1)
    b64s = extract_base64_images(image_blocks)
    if b64s:
        img = b64_to_pil(b64s[idx - 1])
        st.subheader(f"Image {idx}")
        st.image(img, use_column_width=True)
    else:
        st.sidebar.write("_No image payloads found._")
else:
    st.sidebar.write("_No images found._")

# Build or reuse the RAG pipeline
with st.spinner("⚙️ Preparing RAG pipeline..."):
    retrieve_fn, answer_fn = init_rag_pipeline(text_blocks, table_blocks, image_blocks)

# Q&A
st.sidebar.subheader("❓ Ask a Question")
question = st.sidebar.text_input("What do you want to know?")
if question:
    with st.spinner("🤖 Thinking..."):
        answer, context_texts, context_images = ask_question(question, retrieve_fn, answer_fn)
    st.subheader("Answer")
    st.write(answer)

    st.markdown("**Context Texts:**")
    for txt in context_texts:
        st.write(f"- {txt[:200]}…")

    st.markdown("**Context Images (base64 placeholders):**")
    for img_b64 in context_images:
        st.write(f"- {img_b64[:50]}…")
