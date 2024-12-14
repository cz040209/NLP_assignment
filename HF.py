import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism for tokenizers

import streamlit as st
import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer
import torch

# Your Hugging Face token
HF_TOKEN = "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"  # Replace with your token

# Load Summarization Model and Tokenizer
@st.cache_resource
def load_summarization_model():
    model_name = "facebook/bart-large-cnn"  # Replace with your model
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)  # Updated argument
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=HF_TOKEN)  # Updated argument
    return tokenizer, model

# Load MarianMT Model and Tokenizer for Translation (Chinese-English)
@st.cache_resource
def load_translation_model(model_name):
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Initialize models and tokenizers
summarization_tokenizer, summarization_model = load_summarization_model()

# Function to split text into manageable chunks for summarization
def split_text(text, max_tokens=1024):
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
    max_tokens = 1024  # Token limit for the model
    chunks = split_text(text, max_tokens)

    summaries = []
    for chunk in chunks:
        inputs = summarization_tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        summary_ids = summarization_model.generate(
            inputs["input_ids"], max_length=150, min_length=40, num_beams=4, early_stopping=True
        )
        summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    final_summary = " ".join(summaries)
    return final_summary if summaries else "No summary available."

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return " ".join(page.extract_text() for page in pdf.pages if page.extract_text())

# Streamlit App
st.title("Interactive Summarization, Q&A, and Translation Application")
st.subheader("Summarize content from PDFs or manual input, ask questions, and translate text.")

# Option to choose between PDF upload, manual input, or translation
option = st.radio("Choose input method:", ("Upload PDF", "Enter Text Manually"))

# Initialize variables for context
context_text = ""

if option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text:
            st.success("Text extracted successfully!")
            st.text_area("Extracted Text (Preview)", pdf_text[:2000], height=200)
            context_text = pdf_text

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
    manual_text = st.text_area("Enter your text below:", height=200)
    if manual_text.strip():
        context_text = manual_text

        st.subheader("Summarize the Entered Text")
        if st.button("Summarize Text"):
            with st.spinner("Summarizing text..."):
                summary = summarize_text(manual_text)
            st.success("Summary generated!")
            st.write(summary)
    else:
        st.info("Please enter some text to summarize.")

# Translation Section
st.subheader("Translate Text")

# Choose translation direction (English ↔ Chinese)
target_language = st.selectbox("Choose translation direction:", ("English to Chinese", "Chinese to English"))

if context_text:
    st.subheader("Translate the Text")
    if st.button("Translate Text"):
        with st.spinner("Translating text..."):
            if target_language == "English to Chinese":
                model_name = "Helsinki-NLP/opus-mt-en-zh"  # English to Chinese model
            else:
                model_name = "Helsinki-NLP/opus-mt-zh-en"  # Chinese to English model

            translation_model, translation_tokenizer = load_translation_model(model_name)
            
            # Translate the text
            inputs = translation_tokenizer(context_text, return_tensors="pt", padding=True)
            translated = translation_model.generate(**inputs)
            translated_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)

        st.success(f"Translated text ({target_language}):")
        st.write(translated_text)
