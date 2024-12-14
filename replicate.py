import streamlit as st
import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Your Hugging Face token
HF_TOKEN = "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"  # Replace with your token

# Load LLaMA 2 Models and Pipelines
@st.cache_resource
def load_model():
    # Hugging Face model details for LLaMA 2
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Adjust model size as needed

    # Load the tokenizer and model with authentication
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)

    # Create pipelines
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return summarizer, qa_pipeline

# Initialize the pipelines
summarizer, qa_pipeline = load_model()

# Function to summarize text using LLaMA 2
def llama_summarize(text):
    result = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return result[0]['summary_text'] if result else "No summary available."

# Function for Q&A using LLaMA 2
def llama_qa(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer'] if result else "No answer available."

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return " ".join(page.extract_text() for page in pdf.pages)

# Streamlit App
st.title("Interactive Chat Application with PDF (LLaMA 2)")
st.subheader("Upload a PDF to interact with its content using LLaMA 2.")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Extract text from PDF
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)

    if pdf_text:
        st.success("Text extracted successfully!")
        st.text_area("Extracted Text (Preview)", pdf_text[:2000], height=200)

        # Summarize text
        st.subheader("Summarize the PDF Content")
        if st.button("Summarize"):
            with st.spinner("Summarizing text..."):
                summary = llama_summarize(pdf_text)
            st.success("Summary generated!")
            st.write(summary)

        # Q&A Functionality
        st.subheader("Ask Questions About the PDF")
        user_question = st.text_input("Enter your question:")
        if user_question:
            with st.spinner("Processing your question..."):
                answer = llama_qa(user_question, pdf_text)
            st.success("Answer generated!")
            st.write(f"**Answer**: {answer}")
    else:
        st.error("Failed to extract text. Please check your PDF file.")
