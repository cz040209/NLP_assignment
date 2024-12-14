mport streamlit as st
import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Your Hugging Face token
HF_TOKEN = "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"  # Replace with your token

# Load Model and Pipeline
@st.cache_resource
def load_model():
    model_name = "facebook/bart-large-cnn"  # Replace with your model

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)

    # Create summarization pipeline
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    return summarizer

# Initialize the summarizer pipeline
summarizer = load_model()

# Function to split text into manageable chunks
def split_text(text, max_tokens=1024):
    """Splits text into chunks within the token limit."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Function to summarize text
def summarize_text(text):
    """Summarizes text, handling long inputs."""
    max_tokens = 1024  # Token limit for the model
    chunks = split_text(text, max_tokens)

    summaries = []
    for chunk in chunks:
        result = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
        if result:
            summaries.append(result[0]['summary_text'])

    # Combine all chunk summaries
    final_summary = " ".join(summaries)
    return final_summary if summaries else "No summary available."

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return " ".join(page.extract_text() for page in pdf.pages if page.extract_text())

# Streamlit App
st.title("Interactive Chat Application with Text and PDF Summarization")
st.subheader("Summarize content from uploaded PDFs or manually entered text.")

# Option to choose between PDF upload or manual input
option = st.radio("Choose input method:", ("Upload PDF", "Enter Text Manually"))

if option == "Upload PDF":
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
            if st.button("Summarize PDF"):
                with st.spinner("Summarizing text..."):
                    summary = summarize_text(pdf_text)
                st.success("Summary generated!")
                st.write(summary)
        else:
            st.error("Failed to extract text. Please check your PDF file.")

elif option == "Enter Text Manually":
    # Manual text input
    manual_text = st.text_area("Enter your text below:", height=200)

    if manual_text.strip():
        st.subheader("Summarize the Entered Text")
        if st.button("Summarize Text"):
            with st.spinner("Summarizing text..."):
                summary = summarize_text(manual_text)
            st.success("Summary generated!")
            st.write(summary)
    else:
        st.info("Please enter some text to summarize.")
