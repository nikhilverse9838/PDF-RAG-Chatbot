
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI


# -------------------- CONFIGURATION --------------------

load_dotenv()

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)


# -------------------- LOAD MODELS --------------------

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Check your .env file."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        max_tokens=500
    )


def get_response_text(response):
    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)

        return "\n".join(parts).strip()

    return str(content)


# -------------------- PDF PROCESSING --------------------

def process_pdfs(uploaded_files):

    all_documents = []

    for uploaded_file in uploaded_files:

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            # Store the original PDF name in metadata
            for document in documents:
                document.metadata["source"] = uploaded_file.name

            all_documents.extend(documents)

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    if not all_documents:
        raise ValueError("No readable text found in the uploaded PDFs.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_documents)

    embedding_model = load_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vectorstore, len(chunks)


# -------------------- CHAT FUNCTION --------------------

def generate_answer(question):

    vectorstore = st.session_state.vectorstore
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    relevant_docs = retriever.invoke(question)

    if not relevant_docs:
        return "I could not find relevant information in the uploaded PDFs.", []

    context_parts = []

    for doc in relevant_docs:
        filename = doc.metadata.get("source", "Unknown PDF")
        page = doc.metadata.get("page", 0) + 1

        context_parts.append(
            f"Source: {filename}, Page: {page}\n"
            f"Content: {doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    history = ""

    for chat in st.session_state.chat_history[-4:]:
        history += (
            f"User: {chat['question']}\n"
            f"Assistant: {chat['answer']}\n"
        )

    prompt = f"""
You are a helpful PDF-based AI assistant.

Instructions:
1. Answer using the provided PDF context.
2. Do not invent information.
3. If the answer is not available in the context,
   clearly say that it was not found in the uploaded PDFs.
4. Use conversation history to understand follow-up questions.
5. Give clear and easy-to-understand answers.
6. Mention the source PDF and page number when relevant.

Conversation History:
{history if history else "No previous conversation."}

PDF Context:
{context}

Current Question:
{question}

Answer:
"""

    llm = load_llm()
    response = llm.invoke(prompt)
    answer = get_response_text(response)

    sources = []

    for doc in relevant_docs:
        filename = doc.metadata.get("source", "Unknown PDF")
        page = doc.metadata.get("page", 0) + 1

        source = {
            "filename": filename,
            "page": page
        }

        if source not in sources:
            sources.append(source)

    return answer, sources


# -------------------- SESSION STATE --------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


# -------------------- SIDEBAR --------------------

with st.sidebar:

    st.title("📚 PDF Manager")

    uploaded_files = st.file_uploader(
        "Upload your PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("⚙️ Process PDFs", use_container_width=True):

        if not uploaded_files:
            st.warning("Please upload at least one PDF.")

        else:
            try:
                with st.spinner(
                    "Reading PDFs and creating embeddings..."
                ):
                    vectorstore, total_chunks = process_pdfs(
                        uploaded_files
                    )

                    st.session_state.vectorstore = vectorstore
                    st.session_state.chat_history = []
                    st.session_state.processed_files = [
                        file.name for file in uploaded_files
                    ]

                st.success("PDFs processed successfully!")
                st.write("Total chunks:", total_chunks)

            except Exception as error:
                st.error(f"Processing failed: {error}")

    st.divider()

    st.subheader("Processed Documents")

    if st.session_state.processed_files:

        for filename in st.session_state.processed_files:
            st.write(f"📄 {filename}")

    else:
        st.caption("No PDFs processed yet.")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# -------------------- MAIN UI --------------------

st.title("📚 PDF Based RAG Chatbot")

st.markdown(
    "### Ask questions from your documents using AI"
)

st.caption(
    "Upload PDFs, process them, and get answers based on their content."
)

if st.session_state.vectorstore is None:

    st.info(
        "👈 Upload one or more PDF files from the sidebar "
        "and click **Process PDFs** to get started."
    )

    st.markdown("#### How it works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📄")
        st.write("**1. Upload PDF**")
        st.caption("Select your study material or documents.")

    with col2:
        st.markdown("### 🔍")
        st.write("**2. Semantic Search**")
        st.caption("The system finds relevant information.")

    with col3:
        st.markdown("### 🤖")
        st.write("**3. Ask AI**")
        st.caption("Gemini generates answers from PDF context.")

else:

    st.success("Your PDFs are ready! Ask your first question.")

    # Display previous messages
    for chat in st.session_state.chat_history:

        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):
            st.write(chat["answer"])

            if chat["sources"]:
                with st.expander("📌 View Sources"):
                    for source in chat["sources"]:
                        st.write(
                            f"📄 {source['filename']} — "
                            f"Page {source['page']}"
                        )

    # User input
    question = st.chat_input("Ask anything about your PDFs...")

    if question:

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching PDFs and generating answer..."):

                try:
                    answer, sources = generate_answer(question)

                    st.write(answer)

                    if sources:
                        with st.expander("📌 View Sources"):
                            for source in sources:
                                st.write(
                                    f"📄 {source['filename']} — "
                                    f"Page {source['page']}"
                                )

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "sources": sources
                    })

                except Exception as error:
                    st.error(f"Something went wrong: {error}")
