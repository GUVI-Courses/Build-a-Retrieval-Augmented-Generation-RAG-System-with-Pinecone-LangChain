# RAG Chatbot with Pinecone and LangChain

This project implements a Retrieval-Augmented Generation (RAG) chatbot using Pinecone for vector storage and LangChain for document processing and LLM integration.

## Features
- Load and process PDF documents.
- Generate embeddings using OpenAI.
- Store and retrieve embeddings with Pinecone.
- Interactive chatbot interface.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables in `.env`:
   ```env
   OPENAI_API_KEY=your-openai-api-key
   PINECONE_API_KEY=your-pinecone-api-key
   PINECONE_ENVIRONMENT=your-pinecone-environment
   ```

3. Place your PDF file in the `data/` directory.

4. Run the chatbot:
   ```bash
   python main.py
   ```

## Usage
- Ask questions about the content of the PDF.
- Type `exit` or `quit` to end the session.