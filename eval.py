import os
import json
import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

# --- Configuration ---
CSV_PATH = "medium_articles_sample.csv"
EVAL_FILE = "gt.json"
RESULTS_FILE = "complete_grid_results.csv"

EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
LLM_MODEL = "4UHRUIN-gpt-5-mini"

# Grid Parameters
CHUNK_SIZES = [256, 512, 1024]
OVERLAP_RATIOS = [0.05, 0.10, 0.20, 0.30]
TOP_KS = [3, 5, 10, 15]

def setup_environment():
    load_dotenv()
    required = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
    for key in required:
        if not os.environ.get(key):
            raise ValueError(f"Missing {key} in .env file.")
    
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=1)
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    
    return llm, embeddings, index_name

def load_documents():
    print(f"Loading dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    documents = []
    for i, row in df.iterrows():
        title = str(row.get('title', 'Unknown Title'))
        authors = str(row.get('authors', 'Unknown Author'))
        tags = str(row.get('tags', ''))
        text = str(row.get('text', ''))
        article_id = str(row.get('url', f"id_{i}"))

        content = f"Title: {title}\nAuthor: {authors}\nTags: {tags}\nContent: {text}"
        doc = Document(
            page_content=content,
            metadata={"title": title, "article_id": article_id} 
        )
        documents.append(doc)
    return documents

def get_namespace(chunk_size, overlap_ratio):
    return f"c{chunk_size}_o{int(overlap_ratio * 100)}"

def execute_ingestion_phase(documents, embeddings, index_name):
    print(f"\n{'='*50}\nPHASE 1: INGESTING CONFIGURATIONS\n{'='*50}")
    
    for chunk_size in CHUNK_SIZES:
        for overlap_ratio in OVERLAP_RATIOS:
            namespace = get_namespace(chunk_size, overlap_ratio)
                
            overlap_tokens = int(chunk_size * overlap_ratio)
            print(f"[{namespace}] Chunking (Size: {chunk_size}, Overlap: {overlap_tokens})...")
            
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=chunk_size,
                chunk_overlap=overlap_tokens
            )
            chunked_docs = text_splitter.split_documents(documents)
            
            print(f"[{namespace}] Upserting {len(chunked_docs)} chunks...")
            PineconeVectorStore.from_documents(
                documents=chunked_docs,
                embedding=embeddings,
                index_name=index_name,
                namespace=namespace
            )

def run_e2e_pipeline(llm, vectorstore, top_k, raw_query, capability, targets):
    rewrite_prompt = ChatPromptTemplate.from_template(
        "You are an expert search query generator. "
        "Convert the following user question into a concise, keyword-rich search query "
        "optimized for querying a vector database of Medium articles. "
        "Output ONLY the search query, nothing else.\n"
        "User Question: {question}"
    )
    query_rewriter = rewrite_prompt | llm | StrOutputParser()
    optimized_query = query_rewriter.invoke({"question": raw_query})

    results = vectorstore.similarity_search_with_score(optimized_query, k=top_k)
    
    context_text_blocks = []
    retrieved_ids = []
    
    for doc, score in results:
        chunk_text = doc.page_content
        article_id = doc.metadata.get("article_id", doc.metadata.get("url", ""))
        retrieved_ids.append(article_id)
        context_text_blocks.append(chunk_text)

    joined_context = "\n\n---\n\n".join(context_text_blocks)
    
    is_hit = False
    reciprocal_rank = 0.0
    
    if capability != "guardrail":
        for i, r_id in enumerate(retrieved_ids):
            if r_id in targets:
                is_hit = True
                reciprocal_rank = 1.0 / (i + 1)
                break

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

    messages = [
        SystemMessage(content=system_prompt_str),
        HumanMessage(content=raw_query)
    ]
    
    final_response = llm.invoke(messages)
    return is_hit, reciprocal_rank, final_response.content

def execute_evaluation_phase(llm, embeddings, index_name, ground_truth):
    print(f"\n{'='*50}\nPHASE 2: END-TO-END EVALUATION\n{'='*50}")
    
    evaluation_log = []

    for chunk_size in CHUNK_SIZES:
        for overlap_ratio in OVERLAP_RATIOS:
            namespace = get_namespace(chunk_size, overlap_ratio)
            print(f"\nTargeting Namespace: '{namespace if namespace else 'default'}' (Size: {chunk_size}, Overlap: {overlap_ratio})")
            
            vectorstore = PineconeVectorStore(
                index_name=index_name, 
                embedding=embeddings, 
                namespace=namespace
            )
            
            for top_k in TOP_KS:
                print(f"  Testing Top-K: {top_k}")
                
                total_standard_queries = 0
                total_hits = 0
                sum_mrr = 0.0
                
                config_responses = {}

                for item in ground_truth:
                    is_hit, rr, response_text = run_e2e_pipeline(
                        llm=llm,
                        vectorstore=vectorstore,
                        top_k=top_k,
                        raw_query=item["query"],
                        capability=item["capability"],
                        targets=item["target_article_ids"]
                    )
                    
                    config_responses[item["query_id"]] = response_text
                    
                    if item["capability"] != "guardrail":
                        total_standard_queries += 1
                        if is_hit:
                            total_hits += 1
                        sum_mrr += rr

                hit_rate = (total_hits / total_standard_queries) if total_standard_queries > 0 else 0
                mrr = (sum_mrr / total_standard_queries) if total_standard_queries > 0 else 0
                
                log_entry = {
                    "chunk_size": chunk_size,
                    "overlap_ratio": overlap_ratio,
                    "top_k": top_k,
                    "hit_rate": hit_rate,
                    "mrr": mrr
                }
                log_entry.update(config_responses)
                evaluation_log.append(log_entry)
                
    return evaluation_log

def main():
    SKIP_INGESTION = True
    llm, embeddings, index_name = setup_environment()
    
    with open(EVAL_FILE, "r") as f:
        ground_truth = json.load(f)

    if not SKIP_INGESTION:
        documents = load_documents()
        execute_ingestion_phase(documents, embeddings, index_name)
    else:
        print(f"\n{'='*50}\nSKIPPING INGESTION: Using existing databases.\n{'='*50}")

    evaluation_log = execute_evaluation_phase(llm, embeddings, index_name, ground_truth)

    print(f"\n{'='*50}\nPIPELINE COMPLETE. SAVING RESULTS.\n{'='*50}")
    
    df_results = pd.DataFrame(evaluation_log)
    df_results = df_results.sort_values(by=["mrr", "hit_rate"], ascending=[False, False])
    
    print("\nTop 5 Configurations (Sorted by MRR):")
    print(df_results[["chunk_size", "overlap_ratio", "top_k", "hit_rate", "mrr"]].head(5).to_string(index=False))
    
    df_results.to_csv(RESULTS_FILE, index=False)
    print(f"\nFull evaluation metrics and exact LLM responses saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()