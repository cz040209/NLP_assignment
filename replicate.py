import streamlit as st
import pdfplumber
import requests

# Your Replicate API token
REPLICATE_TOKEN = "r8_X3zY9ErRiIIfz5mcNq5hC8LM2ZrYLuN2f2kgW"  # Replace with your token

# Base URL for Replicate API
REPLICATE_BASE_URL = "https://api.replicate.com/v1/predictions"

# Function to make Replicate API call for summarization
def llama_summarize_replicate(text):
    headers = {
        "Authorization": f"Token {REPLICATE_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "version": "version_id_for_llama_summarization",  # Replace with the specific model version
        "input": {"text": text},
    }
    response = requests.post(REPLICATE_BASE_URL, json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
        return result.get("output", "No summary available.")
    else:
        st.error(f"Error {response.status_code}: {response.text}")
        return "Failed to generate summary."

# Function to make Replicate API call for Q&A
def llama_qa_replicate(question, context):
    headers = {
        "Authorization": f"Token {REPLICATE_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "version": "version_id_for_llama_qa",  # Replace with the specific model version
        "input": {"question": question, "context": context},
    }
    response = requests.post(REPLICATE_BASE_URL, json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
        return result.get("output", "No answer available.")
    else:
        st.error(f"Error {response.status_code}: {response.text}")
        return "Failed to generate answer."

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return " ".join(page.extract_text() for page in pdf.pages)

# Streamlit App
st.title("Interactive Chat Application with PDF (Replicate Platform)")
st.subheader("Upload a PDF to interact with its content using LLaMA 2 on Replicate.")

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
                summary = llama_summarize_replicate(pdf_text)
            st.success("Summary generated!")
            st.write(summary)

        # Q&A Functionality
        st.subheader("Ask Questions About the PDF")
        user_question = st.text_input("Enter your question:")
        if user_question:
            with st.spinner("Processing your question..."):
                answer = llama_qa_replicate(user_question, pdf_text)
            st.success("Answer generated!")
            st.write(f"**Answer**: {answer}")
    else:
        st.error("Failed to extract text. Please check your PDF file.")
