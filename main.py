import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from google import generativeai as genai

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ✅ API Key Safety Check
if not GOOGLE_API_KEY:
    st.error("❌ Google API key not found. Please set it in your .env file.")
    st.stop()

# ✅ Configure Gemini SDK
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")  # Use gemini-2.5-flash for speed and cost-efficiency

# ✅ Load Embeddings + VectorStores
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
techport_index = FAISS.load_local("vectorstores/techport_index", embeddings, allow_dangerous_deserialization=True)
techtransfer_index = FAISS.load_local("vectorstores/techtransfer_index", embeddings, allow_dangerous_deserialization=True)

# ✅ Initialize Streamlit App
st.set_page_config(page_title="🚀 NASA Innovation Companion", layout="wide")
st.title("🌌 NASA Innovation Companion")

st.markdown("""
Ask any innovation or technology-related query based on NASA's research and patents.

**Data Sources:**
- 🛰️ TechPort (NASA internal R&D)
- 📜 TechTransfer (NASA patents & spinoffs)

**Example Queries:**
- "How is AI used in space communications?"
- "Any NASA research related to lunar habitats?"
""")

# ✅ Initialize temporary in-session chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Memory limit to prevent context overflow
MAX_CHAT_HISTORY = 6  # keep last 3 interactions
if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
    st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]

# ✅ Input UI
query = st.chat_input("🔍 Ask your question:")

if query:
    with st.spinner("Thinking..."):
        # 🔎 Step 1: Retrieve Relevant Docs
        docs_tp = techport_index.similarity_search(query, k=3)
        docs_tt = techtransfer_index.similarity_search(query, k=3)

        context_tp = "\n\n".join(doc.page_content for doc in docs_tp)
        context_tt = "\n\n".join(doc.page_content for doc in docs_tt)

        # 🔁 Step 2: Convert history into readable format
        chat_history_text = "\n".join(st.session_state.chat_history)

        # 🧠 Step 3: Final prompt
        full_prompt = f"""
You are a helpful and intelligent NASA research assistant. Based on the following documents, provide a detailed, beginner-friendly, and clearly explained answer.

Be descriptive, include analogies or examples if possible. If the user is asking a follow-up question, answer it in the context of the previous chat history.

Documents from TechPort:
{context_tp}

Documents from TechTransfer:
{context_tt}

Chat history (for context):
{chat_history_text}

User question: {query}
"""

        # 💬 Add user message to memory
        st.session_state.chat_history.append(f"User: {query}")

        # 🤖 Step 4: Call Gemini model
        try:
            response = model.generate_content(full_prompt)
            reply = response.text.strip()
        except Exception as e:
            st.error(f"Error: {e}")
            reply = None

        # 📌 Step 5: Store and display response
        if reply:
            st.session_state.chat_history.append(f"NASA Companion: {reply}")
            st.subheader("🧠 Innovation Insights")
            st.write(reply)

        # 🗂️ Step 6: Show Sources
        with st.expander("📄 View Sources (TechPort)"):
            for doc in docs_tp:
                st.markdown(f"**{doc.metadata.get('title', 'No Title')}**\n\n{doc.page_content}")

        with st.expander("📄 View Sources (TechTransfer)"):
            for doc in docs_tt:
                st.markdown(f"**{doc.metadata.get('title', 'No Title')}**\n\n{doc.page_content}")

# 🧠 Display Chat Memory (session only)
if st.session_state.chat_history:
    with st.expander("🧠 View Temporary Chat Memory"):
        for msg in st.session_state.chat_history:
            if msg.startswith("User:"):
                st.markdown(f"**🧑 You:** {msg.replace('User: ', '', 1)}")
            elif msg.startswith("NASA Companion:"):
                st.markdown(f"**🤖 NASA Companion:** {msg.replace('NASA Companion: ', '', 1)}")
