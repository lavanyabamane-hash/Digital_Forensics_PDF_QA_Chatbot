import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# ---------------------------------
# Streamlit Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Digital Forensics PDF QA Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Digital Forensics PDF QA Chatbot")
st.write("Upload one or more PDF files and ask questions about their contents.")


# ---------------------------------
# Sidebar
# ---------------------------------
with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "Enter your Gemini API Key",
        type="password"
    )

    uploaded_files = st.file_uploader(
        "Upload PDF File(s)",
        type="pdf",
        accept_multiple_files=True
    )


# ---------------------------------
# Set API Key
# ---------------------------------
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key


# ---------------------------------
# Process PDFs
# ---------------------------------
if uploaded_files and api_key:

    # Process PDFs only once
    if "vector_db" not in st.session_state:

        with st.spinner("Processing PDF(s)..."):

            documents = []

            for uploaded_file in uploaded_files:

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name

                loader = PyPDFLoader(temp_path)
                documents.extend(loader.load())

            # Split Documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            docs = text_splitter.split_documents(documents)

            # Embedding Model
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # Create FAISS Database
            st.session_state.vector_db = FAISS.from_documents(
                docs,
                embedding_model
            )

        st.success("✅ PDF(s) processed successfully!")

    # ---------------------------------
    # Gemini Model
    # ---------------------------------
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key
    )

    # ---------------------------------
    # Ask Question
    # ---------------------------------
    question = st.text_input("Ask a question about your PDF(s)")

    if question:

        with st.spinner("Searching..."):

            docs = st.session_state.vector_db.similarity_search(
                question,
                k=3
            )

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            prompt = f"""
You are an expert Digital Forensics assistant.

Answer ONLY using the context below.

If the answer is not present in the context, reply:
"I couldn't find that information in the uploaded PDF."

Context:
{context}

Question:
{question}

Answer:
"""

            response = llm.invoke(prompt)

        # ---------------------------------
        # Display Answer
        # ---------------------------------
        st.subheader("📌 Answer")

        answer = ""

        if isinstance(response.content, str):
            answer = response.content

        elif isinstance(response.content, list):
            for item in response.content:

                if isinstance(item, dict):
                    if item.get("type") == "text":
                        answer += item.get("text", "")

                elif hasattr(item, "text"):
                    answer += item.text

                else:
                    answer += str(item)

        else:
            answer = str(response.content)

        st.success(answer)

        # ---------------------------------
        # Show Retrieved Sources
        # ---------------------------------
        with st.expander("📄 Retrieved Context"):

            for i, doc in enumerate(docs, start=1):
                st.markdown(f"### Source {i}")
                st.write(doc.page_content)
                st.divider()
