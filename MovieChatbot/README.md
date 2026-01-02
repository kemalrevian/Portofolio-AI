# 🎬 Movie Recommendation Chatbot
An AI-powered movie expert chatbot leveraging an Agentic Retrieval-Augmented Generation (RAG) approach, integrating Qdrant-based vector search and web search APIs to provide reliable movie recommendations

## 🚀 Key Features
- Agentic RAG (ReAct-style)
  1. Uses a ReAct-based agent to reason internally before selecting tools
- Vector-Based Movie Retrieval
  1. Movie plots embedded and stored in Qdrant Vector Database
  2. Semantic search over Wikipedia movie plot data
  3. Cross-encoder reranking to improve retrieval quality
- Tool-Oriented Design
  1. Vector DB retrieval for plot-based and genre-based queries
  2. Web search tool for ratings, box office, and release dates
  3. Enforced tool usage based on query intent

## 🛠 Tech Stack
- Python
- LangChain
- LangGraph
- OpenAI
- Qdrant
- SearchAPI
- Streamlit

