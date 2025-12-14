import os
import pandas as pd
from uuid import uuid4
from dotenv import load_dotenv
import openai
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient

# =====================
# 1. LOAD ENV
# =====================
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
api_key_qdrant = os.getenv("QDRANT_API_KEY")

QDRANT_URL = "https://4c8e090e-7568-43ba-abe2-7c63b5df3402.eu-central-1-0.aws.cloud.qdrant.io"
COLLECTION_NAME = "movies"

EMBEDDING_MODEL = "text-embedding-3-small"

# =====================
# 2. SETUP EMBEDDINGS
# =====================
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=openai.api_key
)

# =====================
# 3. LOAD & CLEAN DATA
# =====================
df = pd.read_csv("wiki_movie_plots_deduped.csv")

data = df.copy()

data = data.dropna(subset=['Title', 'Genre', "Plot"]).reset_index(drop=True) #delete rows that contain missing values
data = data.drop_duplicates(subset=['Title', 'Genre', "Plot"]).reset_index(drop=True) #delete duplicate rows
# data.head()
data = data.dropna(subset=['Title', 'Genre', "Plot"])


# =====================
# 4. CREATE DOCUMENT
# =====================
documents = []
for i in range(data.shape[0]):
  title = data["Title"][i]
  genres = data["Genre"][i]
  Plot = data["Plot"][i]
  doc = Document(
      page_content=f"{title}\n{Plot}",
      metadata={"Title": str(title), "genres": str(genres),"Plot": str(Plot)},
  )
  documents.append(doc)

#setup unique id
uuids = [str(uuid4()) for _ in range(len(documents))]
# print(documents[0])

# ===================================
# 5. SAVE TO QDRANT (Vector Database)
# ==================================
url_qdrant = "https://4c8e090e-7568-43ba-abe2-7c63b5df3402.eu-central-1-0.aws.cloud.qdrant.io"
qdrant = QdrantVectorStore.from_documents(
    documents,
    embeddings,
    url=url_qdrant,
    prefer_grpc=False,
    api_key=api_key_qdrant,
    collection_name="movies",
)

# ======================
# 6. SETUP QDRANT CLIENT
# ======================
url_qdrant = "https://4c8e090e-7568-43ba-abe2-7c63b5df3402.eu-central-1-0.aws.cloud.qdrant.io"
client = QdrantClient(
  url= url_qdrant,
  api_key = api_key_qdrant
)
#get all collection in your qdrant vector database
collections_response = client.get_collections()
print("Collections:", collections_response.collections)