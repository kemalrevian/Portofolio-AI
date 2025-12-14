import streamlit as st
import tempfile
import os

from core import (
    extract_text_from_pdf,
    chunking_long_text,
    map_summarization,
    reduce_summarization
)

# ============
# Page Config
# ============
st.set_page_config(
    page_title="📄 PDF Summarizer",
    page_icon="📄",
    layout="centered"
)

st.title("📄 AI PDF Summarizer")
st.caption("Upload a PDF file and get an AI-generated summary")

# ===========
# File Upload
# ===========
uploaded_file = st.file_uploader(
    "Upload PDF file",
    type=["pdf"]
)

# ===================
# Summarization Logic
# ===================
if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("🔍 Summarize"):
        with st.spinner("Reading and summarizing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name

            try:
                # Extract text
                text = extract_text_from_pdf(pdf_path)
                # Chunking
                chunks = chunking_long_text(text)
                # Map
                map_results = map_summarization(chunks)
                # Reduce
                final_summary = reduce_summarization(map_results)

                st.subheader("📌 Summary")
                st.write(final_summary)

            finally:
                os.remove(pdf_path)
