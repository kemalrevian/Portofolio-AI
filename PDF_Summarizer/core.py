from dotenv import load_dotenv
import os
import openai
from langchain_openai import ChatOpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from pypdf import PdfReader

# ==============
# ENV and Config
# ==============
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0
)

# ============
# Extract Text
# ============
def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text

# =============
# Chunking Text
# =============
def chunking_long_text(content):
    splitter = RecursiveCharacterTextSplitter(
        # separators=[".", " "],
        chunk_size = 1000,
        chunk_overlap=10
    )
    chunks = splitter.split_text(content)
    return chunks

# =================
# Map Summarization
# =================
map_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    You are a helpful assistant.
    Summarize the following text clearly and concisely.
    Focus on key ideas and important facts.

    TEXT:
    {text}
"""
)

map_chain = map_prompt | llm

def map_summarization(chunks):
    summaries = []

    for chunk in chunks:
        summary = map_chain.invoke({"text": chunk})
        summaries.append(summary.content)
    return summaries


# ====================
# Reduce summarization
# ====================
reduce_prompt = PromptTemplate.from_template(
    """
You are a summarization assistant.

Combine the following partial summaries into ONE coherent, well-structured summary.
Avoid repetition and preserve important details.

SUMMARIES:
{summaries}
"""
)

reduce_chain = reduce_prompt | llm

def reduce_summarization(summaries):
    joined_summaries = "\n".join(summaries)
    final_summary = reduce_chain.invoke({"summaries": joined_summaries})

    return final_summary.content