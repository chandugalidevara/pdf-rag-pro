import streamlit as st
import os
import tempfile
import shutil
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from fpdf import FPDF

st.set_page_config(page_title="PDF RAG Pro", layout="wide", page_icon="📄")

st.title("📄 PDF RAG Pro - Near-Zero Hallucination")
st.caption("FlashRank Reranker + LlamaParse + Accurate Page Numbers")

load_dotenv(override=True)

GROQ_API_KEY = "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"   # ← Your Groq key
LLAMA_CLOUD_API_KEY = "llx-cWhSAtPyFPCYRZlFJCUBY5wKYpB8XucNDZnaErtnxXKAmr26"               # ← Your LlamaParse key

if not GROQ_API_KEY or GROQ_API_KEY.startswith("gsk_PASTE"):
    st.error("Please update your Groq API key")
    st.stop()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ Settings")
    use_llamaparse = st.checkbox("Use LlamaParse (for scanned PDFs)", value=False)
    
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if st.button("🚀 Process PDFs & Build Index", type="primary"):
        if not uploaded_files:
            st.error("Upload at least one PDF")
            st.stop()

        if os.path.exists("./chroma_db"):
            try:
                shutil.rmtree("./chroma_db", ignore_errors=True)
            except:
                pass
        st.session_state.vectorstore = None
        st.session_state.messages = []

        with st.spinner("Processing PDFs..."):
            all_docs = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                if use_llamaparse and LLAMA_CLOUD_API_KEY:
                    loader = LlamaParse(api_key=LLAMA_CLOUD_API_KEY, result_type="markdown")
                    llama_docs = loader.load_data(tmp_path)
                    for d in llama_docs:
                        # Accurate page number for LlamaParse
                        page_num = d.metadata.get("page_label") or d.metadata.get("page") or d.metadata.get("page_number") or "1"
                        doc = Document(
                            page_content=d.text,
                            metadata={"source": uploaded_file.name, "page": page_num}
                        )
                        all_docs.append(doc)
                else:
                    loader = PyMuPDFLoader(tmp_path)
                    docs = loader.load()
                    for doc in docs:
                        page_num = doc.metadata.get("page", 0) + 1
                        doc.metadata["source"] = uploaded_file.name
                        doc.metadata["page"] = page_num
                    all_docs.extend(docs)

                os.unlink(tmp_path)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=180)
            splits = text_splitter.split_documents(all_docs)
            splits = [s for s in splits if len(s.page_content.strip()) > 30]

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vectorstore = Chroma.from_documents(
                documents=splits, embedding=embeddings, persist_directory="./chroma_db"
            )
            st.success(f"✅ {len(splits)} chunks indexed from {len(uploaded_files)} PDFs")

    if st.button("🔄 Clear Index"):
        if os.path.exists("./chroma_db"):
            try:
                shutil.rmtree("./chroma_db", ignore_errors=True)
            except:
                pass
        st.session_state.vectorstore = None
        st.session_state.messages = []
        st.success("Everything cleared")

if st.session_state.vectorstore is None:
    st.info("👈 Upload PDFs → click Process")
else:
    base_retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 8})
    compressor = FlashrankRerank(top_n=4)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, api_key=GROQ_API_KEY)

    system_prompt = """Answer ONLY using the provided context. 
If the answer is not in the context, reply exactly: "I don't have sufficient information in the provided documents."
Always cite sources as [Source: filename - Page X]."""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    def format_docs(docs):
        return "\n\n---\n\n".join([f"Source: {d.metadata['source']} - Page {d.metadata.get('page', '?')}\n{d.page_content}" for d in docs])

    rag_chain = (
        {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask anything about the PDFs..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = rag_chain.invoke(question)
                st.markdown(answer)

                with st.expander("📚 Sources & Citations", expanded=True):
                    retrieved = compression_retriever.invoke(question)
                    for i, d in enumerate(retrieved):
                        source = d.metadata.get('source', 'Unknown')
                        page = d.metadata.get('page', '?')
                        snippet = d.page_content.strip()[:480]
                        if len(d.page_content) > 480:
                            snippet += "..."
                        snippet = snippet.replace('\n', ' ').replace('\r', ' ')
                        st.markdown(f"**[{i+1}] {source} - Page {page}**")
                        st.caption(snippet)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.button("📥 Download Chat as PDF"):
        if st.session_state.messages:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="PDF RAG Chat History", ln=True, align='C')
            pdf.ln(10)
            for msg in st.session_state.messages:
                role = "You" if msg["role"] == "user" else "Assistant"
                pdf.multi_cell(0, 8, txt=f"{role}: {msg['content']}")
                pdf.ln(5)
            pdf_bytes = bytes(pdf.output(dest="S"))
            st.download_button(
                label="Click to Download PDF",
                data=pdf_bytes,
                file_name=f"PDF_RAG_Chat_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf"
            )

st.caption("Final Version | Accurate Page Numbers Fixed")