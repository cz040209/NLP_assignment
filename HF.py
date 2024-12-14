import streamlit as st
import pdfplumber
from transformers import pipeline

# Hugging Face token
HF_TOKEN = "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"  # Replace with your token

# Load pipelines with caching
@st.cache_resource
def load_pipelines():
    # Load summarization and question-answering pipelines
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return summarizer, qa_pipeline

# Initialize the pipelines
summarizer, qa_pipeline = load_pipelines()

# Function to summarize text
def summarize_text(text):
    result = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return result[0]['summary_text'] if result else "No summary available."

# Function for Q&A
def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer'] if result else "No answer available."

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return " ".join(page.extract_text() for page in pdf.pages if page.extract_text())

# Streamlit App
st.title("Interactive PDF Chat Application")
st.subheader("Upload a PDF to summarize and interact with its content.")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Extract text from PDF
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)

    if pdf_text:
        st.success("Text extracted successfully!")
        st.text_area("Extracted Text (Preview)", pdf_text[:2000], height=200)

        # Summarization functionality
        st.subheader("Summarize the PDF Content")
        if st.button("Summarize"):
            with st.spinner("Summarizing text..."):
                summary = summarize_text(pdf_text)
            st.success("Summary generated!")
            st.write(summary)

        # Q&A functionality
        st.subheader("Ask Questions About the PDF")
        user_question = st.text_input("Enter your question:")
        if user_question:
            with st.spinner("Processing your question..."):
                answer = answer_question(user_question, pdf_text)
            st.success("Answer generated!")
            st.write(f"**Answer**: {answer}")
    else:
        st.error("Failed to extract text. Please check your PDF file.")
