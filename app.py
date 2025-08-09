import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(page_title="PDF Summarizer & Q&A", layout="wide")
st.title("📚 PDF Summarizer & Q&A Assistant:")

# Load summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Helper function: extract text
def extract_text_from_pdf(uploaded_file):
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    return text

# Upload file
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")



if uploaded_file:
    st.success("PDF uploaded successfully.")
    full_text = extract_text_from_pdf(uploaded_file)
    # print(full_text)

    # Summarization
    if st.button("Summarize Document"):
        chunks = [full_text[i:i+1000] for i in range(0, len(full_text), 1000)]
        summary = ""
        for chunk in chunks:
            summary += summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] + " "
        st.subheader("📄 Summary")
        st.write(summary)

    # Q&A mode
    st.subheader("🤖 Ask Questions About the PDF")
    query = st.text_input("Type your question")

    if query:
        sentences = full_text.split(". ")
        corpus_embeddings = embedder.encode(sentences)
        index = faiss.IndexFlatL2(384)
        index.add(np.array(corpus_embeddings))

        query_embedding = embedder.encode([query])
        D, I = index.search(np.array(query_embedding), k=3)
        top_k_sentences = [sentences[i] for i in I[0]]

        context = " ".join(top_k_sentences)
        qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        answer = qa_pipeline(question=query, context=context)

        st.markdown(f"**Answer:** {answer['answer']}")
