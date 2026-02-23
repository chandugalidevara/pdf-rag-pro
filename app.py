import streamlit as st
import os
import tempfile
import shutil
from datetime import datetime
from dotenv import load_dotenv
import json

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
st.caption("FlashRank + LlamaParse + Enhanced Features")

load_dotenv(override=True)

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
LLAMA_CLOUD_API_KEY = st.secrets.get("LLAMA_CLOUD_API_KEY")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Theme Toggle
if st.button("🌙 Toggle Dark/Light Mode"):
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

if st.session_state.theme == "light":
    st.markdown("""
    <style>
        .stApp { background-color: #ffffff; color: #000000; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .stApp { background-color: #0e1117; color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

# Voice Input (Browser-based)
if st.button("🎤 Speak Question"):
    st.write("Voice input is coming soon! For now, please type your question.")

# Save / Load Chat History
col1, col2 = st.columns(2)
with col1:
    if st.button("💾 Save Chat"):
        with open("chat_history.json", "w") as f:
            json.dump(st.session_state.messages, f)
        st.success("Chat saved!")

with col2:
    if st.button("📂 Load Previous Chat"):
        if os.path.exists("chat_history.json"):
            with open("chat_history.json", "r") as f:
                st.session_state.messages = json.load(f)
            st.success("Chat loaded!")

# Rest of the app (stable version)
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
                        st.markdown(f"**[{i+1}] {d.metadata.get('source', 'Unknown')} - Page {d.metadata.get('page', '?')}**")
                        st.caption(d.page_content[:480] + "...")

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

st.caption("Enhanced Version with Dark/Light Mode + Voice + History")