import os
from dotenv import load_dotenv
from typing import List
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from sentence_transformers import CrossEncoder
from langchain_community.utilities import SearchApiAPIWrapper




# =====================
# 1. ENV & CONFIG
# =====================
load_dotenv()

SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
api_key_qdrant = os.getenv("QDRANT_API_KEY")

QDRANT_URL = "https://4c8e090e-7568-43ba-abe2-7c63b5df3402.eu-central-1-0.aws.cloud.qdrant.io"
COLLECTION_NAME = "movies"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

RERANK_MODEL = "mixedbread-ai/mxbai-rerank-large-v1"


# =====================
# 2. MODELS
# =====================
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model=CHAT_MODEL)


# =====================
# 3. QDRANT SETUP
# =====================
url_qdrant = "https://4c8e090e-7568-43ba-abe2-7c63b5df3402.eu-central-1-0.aws.cloud.qdrant.io"
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="movies",
    url=url_qdrant,
    api_key=api_key_qdrant,
)

# =====================
# 4. RERANKER
# =====================
rerank_model = CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1")

def rerank_documents(query, retrieved_docs, top_k):
    docs_text = [doc.page_content for doc in retrieved_docs]

    reranked_results = rerank_model.rank(
        query=query,
        documents=docs_text,
        return_documents=False,
        top_k = top_k
    )
    return reranked_results    


# =====================
# 5. TOOLS
# =====================
@tool
def search_web(query: str):
  """
    Search movie information such as ratings, box office, release date.
  """
  search = SearchApiAPIWrapper()
  return search.run(query)

@tool
def qdrant_retriever(query:str):
    """
    FUngsi untuk mengambil konteks dari vector DB yang berisi embedding Wikipedia Movie Plots.
    """
    # Step 1 — Ambil kandidat dari Qdrant
    # k besar agar reranker mendapat banyak pilihan
    retrieved = qdrant.similarity_search(
        query=query,
        k=15
    )

    # Step 2 — Rerank dengan cross-encoder
    ranked_docs = rerank_documents(
        query=query,
        retrieved_docs=retrieved,
        top_k=3   # final documents untuk LLM
    )
    return ranked_docs

# =====================
# 6. AGENT
# =====================
PROMPT_MESSAGES = SystemMessage(content="""
                                    Anda adalah Movie Expert Agent dengan kemampuan Agentic RAG.

                                    DOMAIN BATASAN (WAJIB):
                                    Anda HANYA boleh menjawab pertanyaan yang berkaitan dengan:
                                    - film / movie
                                    - rekomendasi film
                                    - genre film
                                    - aktor dan sutradara
                                    - rating, box office, dan tahun rilis film
                                
                                    Gunakan langkah berikut:
                                    1. Tentukan apakah perlu:
                                    - retrieval ke vector DB (qdrant_retriever)
                                        Jika menggunakan Vector DB, 
                                        Berikan jawaban terstruktur dengan format:

                                        **Judul Film:**  
                                        **Genre:**  
                                        **Ringkasan / Penjelasan:** 
                                        
                                    - web search (search_web)
                                        Jika pertanyaan menyebut "rating", "Rotten Tomatoes", "IMDb", atau "box office"
                                            ANDA WAJIB memanggil tool search_web.
                                           Jika tool gagal, katakan "data tidak ditemukan".
                                    - atau keduanya.

                                    2. Jika pertanyaan soal:
                                    - "film tentang X" → gunakan qdrant_retriever
                                    - "rating", "box office", "rilis kapan" → gunakan search_web
                                    - gabungan (contoh: "film X dan rating Rotten Tomatoes-nya") → gunakan keduanya.

                                    3. Selalu lakukan reasoning internal sebelum memilih tool.
                                    4. Jangan tampilkan reasoning ke user.
                                    5. JIKA USER BERTANYA DI LUAR DOMAIN:
                                        - Jangan menjawab kontennya
                                        - Tolak dengan sopan
                                        - Arahkan kembali ke topik film
                                    Jawablah menggunakan hasil tool.
""")


def build_agent(): 
    store_agent = create_react_agent(
        model="openai:gpt-4o-mini",
        tools=[search_web,qdrant_retriever],
        prompt=PROMPT_MESSAGES
    )
    return store_agent