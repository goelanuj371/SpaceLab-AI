# scripts/embed_techtransfer_data.py

import os
import requests
import time
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
NASA_API_KEY = os.getenv("NASA_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# === 1. Fetch TechTransfer Patent Data from NASA API ===
def fetch_techtransfer_data(query="robotics"):
    url = f"https://api.nasa.gov/techtransfer/patent/?{query}&api_key={NASA_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["results"]
    else:
        raise Exception(f"TechTransfer API Error: {response.status_code} - {response.text}")

# === 2. Transform to LangChain Documents ===
def convert_to_documents(results):
    documents = []
    for entry in results:
        # Each entry is a list (not dict), with known index structure
        title = entry[1]
        description = entry[3]
        url = entry[10] if len(entry) > 10 else "N/A"

        content = f"Title: {title}\n\nDescription: {description}"
        metadata = {
            "source": "TechTransfer API",
            "url": url
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

# === 3. Embed and Save ===
def embed_documents(documents):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("vectorstores/techtransfer_index")
    print("✅ TechTransfer data embedded and saved to FAISS.")

# === Main Execution ===
if __name__ == "__main__":
    print("🚀 Fetching TechTransfer data...")
    results = fetch_techtransfer_data("robotics")  # Feel free to change query
    print(f"🔍 {len(results)} entries retrieved.")
    
    print("🛠 Converting to LangChain Documents...")
    docs = convert_to_documents(results)
    
    print("📦 Embedding and saving to FAISS...")
    embed_documents(docs)
