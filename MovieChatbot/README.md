# 🎬 Movie Recommendation Chatbot
An AI-powered movie expert chatbot leveraging an Agentic Retrieval-Augmented Generation (RAG) approach, integrating Qdrant-based vector search and web search APIs to provide reliable movie recommendations

## 🚀 Key Features
- Key Features:
Agentic RAG (ReAct-style)
Uses a ReAct-based agent to reason internally before selecting tools
- Vector-Based Movie Retrieval
Movie plots embedded and stored in Qdrant Vector Database
Semantic search over Wikipedia movie plot data
Cross-encoder reranking to improve retrieval quality
- Tool-Oriented Design
Vector DB retrieval for plot-based and genre-based queries
Web search tool for ratings, box office, and release dates
Enforced tool usage based on query intent

## 🛠 Tech Stack
- Python
- LangChain
- LangGraph
- OpenAI
- Qdrant
- SearchAPI
- Streamlit

