# scripts/embed_techport_data.py

import pandas as pd
import os
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Load CSV
df = pd.read_csv("data/NASA_TechPort.csv")

# Drop rows with missing values in key columns
df.dropna(subset=["Project Title", "Project Description"], inplace=True)

# Create Document objects
documents = []
for _, row in df.iterrows():
    content = f"Title: {row['Project Title']}\n\nDescription: {row['Project Description']}"
    metadata = {
        "id": row["TechPort ID"],
        "taxonomy": row["Primary Taxonomy"],
        "url": row["Project URL"],
        "program": row["Responsible NASA Program"]
    }
    documents.append(Document(page_content=content, metadata=metadata))

# Embed and save to FAISS
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("vectorstores/techport_index")

print("✅ TechPort data embedded and saved to FAISS.")
