import asyncio
import aiohttp
import faiss
import pickle
import re
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the FAISS index and chunks
try:
    index = faiss.read_index("faiss_index.bin")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
except (FileNotFoundError, pickle.UnpicklingError, faiss.FaissException) as e:
    print(f"Error loading data: {e}")
    chunks = []
    index = None

# Define asynchronous API call function to handle the requests efficiently
async def call_deepseek_async(query, context):
    payload = {
        "model": "llama-3-8b-gpt-4o-ru1.0",  # Update with the correct model name
        "messages": [
            {"role": "system", "content": "Vui lòng trả lời bằng tiếng Việt."},
            {"role": "system", "content": "Dựa trên thông tin sau: " + "\n".join(context)},
            {"role": "user", "content": query}
        ]
    }

    try:
        # Use aiohttp to send async requests
        async with aiohttp.ClientSession() as session:
            async with session.post("http://localhost:1234/v1/chat/completions", json=payload, timeout=30) as response:
                response.raise_for_status()
                if response.status == 200:
                    raw_answer = await response.json()
                    clean_answer = re.sub(r'<think>.*?</think>', '', raw_answer.get("choices", [{}])[0].get("message", {}).get("content", ""), flags=re.DOTALL).strip()
                    return clean_answer
                else:
                    print(f"API request failed with status code {response.status}")
                    return "Error: Unable to retrieve response."
    except aiohttp.ClientError as e:
        print(f"Request error: {e}")
        return "Error: Unable to reach the API server."

# Use asyncio to handle multiple requests simultaneously
def call_deepseek(query, context):
    return asyncio.run(call_deepseek_async(query, context))

# RAG search function to find the relevant context based on the query
def rag_search(query):
    if index is None or not chunks:
        return "Error: FAISS index or chunks data is missing."
    
    query_embedding = embed_model.encode([query])
    D, I = index.search(query_embedding, k=3)  # Retrieve top-3 most similar contexts
    context = [chunks[i] for i in I[0]]
    answer = call_deepseek(query, context)
    return answer, context

# Function to inspect the FAISS search results
def inspect_faiss(query):
    if index is None or not chunks:
        return "Error: FAISS index or chunks data is missing."
    
    query_embedding = embed_model.encode([query])
    D, I = index.search(query_embedding, k=3)  # Retrieve top-3 results
    return {
        "query": query,
        "indices": I[0].tolist(),
        "distances": D[0].tolist(),
        "contexts": [chunks[i] for i in I[0]]
    }

# Function to get all available chunks
def get_all_chunks():
    return chunks

# Function to get the metadata of the vector database
def get_vector_db_info():
    if index is None:
        return {
            "error": "FAISS index is missing."
        }
    return {
        "total_chunks": len(chunks),
        "vector_dimension": index.d
    }
