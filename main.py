import os
from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec

# --- Pinecone client ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "ai-rag-chatbot"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# --- Load & split PDF ---
loader = PyPDFLoader("data\👉 How To Break Free From Past Pain 💔 Dr. Bessel van der Kolk Explain Clearly.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# --- Embeddings ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

# --- VectorStore (direct insert) ---
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name
)

# --- Retriever & QA ---
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=OPENAI_API_KEY),
    retriever=retriever
)

print("RAG Chatbot is ready! Ask your questions.")
while True:
    q = input("You: ")
    if q.lower() in ["exit", "quit"]:
        break
    print("Bot:", qa_chain.run(q))
