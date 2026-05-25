import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

# --- Configuration ---
EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
LLM_MODEL = "4UHRUIN-gpt-5-mini"

CHUNK_SIZE = 256
OVERLAP_RATIO = 0.3
TOP_K = 5
NAMESPACE = "production"

load_dotenv()

required_keys = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
for key in required_keys:
    if not os.environ.get(key):
        raise ValueError(f"{key} is missing. Check your .env file.")

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model=LLM_MODEL, temperature=1)
index_name = os.environ.get("PINECONE_INDEX_NAME")

vectorstore = PineconeVectorStore(
    index_name=index_name, 
    embedding=embeddings,
    namespace=NAMESPACE
)

app = FastAPI()

class PromptRequest(BaseModel):
    question: str

@app.post("/api/prompt")
async def run_prompt(request: PromptRequest):
    raw_query = request.question
    
    if not raw_query.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    rewrite_template = (
        "You are an expert search query generator. "
        "Convert the following user question into a concise, keyword-rich search query "
        "optimized for querying a vector database of Medium articles. "
        "Output ONLY the search query, nothing else.\n"
        "User Question: {question}"
    )
    rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)
    query_rewriter = rewrite_prompt | llm | StrOutputParser()
    optimized_query = query_rewriter.invoke({"question": raw_query})

    results = vectorstore.similarity_search_with_score(optimized_query, k=TOP_K)
    
    context_output = []
    context_text_blocks = []
    
    for i, (doc, score) in enumerate(results):
        chunk_text = doc.page_content
        article_id = doc.metadata.get("article_id", doc.metadata.get("url", f"local_id_{i}")) 
        title = doc.metadata.get("title", "Unknown Title")
        
        context_output.append({
            "article_id": article_id,
            "title": title,
            "chunk": chunk_text,
            "score": float(score) 
        })
        context_text_blocks.append(chunk_text)

    joined_context = "\n\n---\n\n".join(context_text_blocks)

    system_prompt_str = (
        "You are a Medium-article assistant that answers questions strictly and only "
        "based on the Medium articles dataset context provided to you (metadata and article passages). "
        "You must not use any external knowledge, the open internet, or information that is not "
        "explicitly contained in the retrieved context. If the answer cannot be determined from "
        "the provided context, respond: \"I don't know based on the provided Medium articles data.\" "
        "Always explain your answer using the given context, quoting or paraphrasing the relevant "
        "article passage or metadata when helpful.\n\n"
        f"Context:\n{joined_context}"
    )
    user_prompt_str = raw_query

    messages = [
        SystemMessage(content=system_prompt_str),
        HumanMessage(content=user_prompt_str)
    ]
    
    final_response = llm.invoke(messages)

    return {
        "response": final_response.content,
        "context": context_output,
        "Augmented_prompt": {
            "System": system_prompt_str,
            "User": user_prompt_str
        }
    }

@app.get("/api/stats")
async def get_stats():
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }