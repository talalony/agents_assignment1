import os
import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# --- Configuration ---
CSV_PATH = "medium-english-50mb.csv"
EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"

CHUNK_SIZE = 256
OVERLAP_RATIO = 0.30
OVERLAP_TOKENS = int(CHUNK_SIZE * OVERLAP_RATIO)
NAMESPACE = "production"

def setup_environment():
    load_dotenv()
    required = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
    for key in required:
        if not os.environ.get(key):
            raise ValueError(f"Missing {key} in .env file.")
    
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    return embeddings, index_name

def load_documents():
    print(f"Loading full dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    documents = []
    for i, row in df.iterrows():
        title = str(row.get('title', 'Unknown Title'))
        authors = str(row.get('authors', 'Unknown Author'))
        tags = str(row.get('tags', ''))
        text = str(row.get('text', ''))
        article_id = str(row.get('url', f"id_{i}"))

        # Skip rows with no text
        if not text.strip() or text.lower() == 'nan':
            continue

        content = f"Title: {title}\nAuthor: {authors}\nTags: {tags}\nContent: {text}"
        doc = Document(
            page_content=content,
            metadata={"title": title, "article_id": article_id} 
        )
        documents.append(doc)
    
    print(f"Loaded {len(documents)} articles.")
    return documents

def main():
    embeddings, index_name = setup_environment()
    documents = load_documents()

    print(f"\nSplitting documents (Chunk Size: {CHUNK_SIZE}, Overlap: {OVERLAP_TOKENS})...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP_TOKENS
    )
    
    chunked_docs = text_splitter.split_documents(documents)
    print(f"Generated {len(chunked_docs)} total chunks.")
    
    PineconeVectorStore.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        index_name=index_name,
        namespace=NAMESPACE
    )
    
    print("\nFull dataset ingestion complete.")

if __name__ == "__main__":
    main()