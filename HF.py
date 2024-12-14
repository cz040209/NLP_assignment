import os
import streamlit as st
import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer
import torch
import speech_recognition as sr  # For audio-to-text functionality
from PIL import Image
import pytesseract  # For OCR (Image to Text)

# Set the path to the tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\chenz\Downloads\tesseract-ocr-w64-setup-5.5.0.20241111\tesseract.exe'

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

# Function for Audio-to-Text (Speech Recognition)
def audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_file)
    with audio as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)

# Function for Image to Text (OCR)
def image_to_text(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

# History storage - will store interactions as tuples (user_input, response_output)
if 'history' not in st.session_state:
    st.session_state.history = []

# Streamlit App
st.title("Interactive Summarization, Q&A, and Translation Application")
st.subheader("Summarize content from PDFs, manual input, ask questions, translate text, and process multimedia!")

# Display an interactive input method for the user
st.subheader("Input your content:")
input_method = st.selectbox("Choose input method:", ["Enter Text", "Upload PDF", "Upload Audio", "Upload Image"])

context_text = ""

# Handle different input methods based on the user's choice
if input_method == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text:
            st.success("Text extracted successfully!")
            st.text_area("Extracted Text (Preview)", pdf_text[:2000], height=200)
            context_text = pdf_text

elif input_method == "Enter Text":
    manual_text = st.text_area("Enter your text below:", height=200)
    if manual_text.strip():
        context_text = manual_text
        if st.button("Summarize Text"):
            with st.spinner("Summarizing text..."):
                summary = summarize_text(manual_text)
            st.success("Summary generated!")
            st.write(summary)
            st.session_state.history.append(("Manual Text", summary))

elif input_method == "Upload Audio":
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])
    if audio_file:
        with st.spinner("Transcribing audio to text..."):
            try:
                transcription = audio_to_text(audio_file)
                st.success("Transcription successful!")
                st.write(transcription)
                st.session_state.history.append(("Audio Upload", transcription))
            except Exception as e:
                st.error(f"Error: {e}")

elif input_method == "Upload Image":
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if image_file:
        with st.spinner("Extracting text from image..."):
            image_text = image_to_text(image_file)
            st.success("Text extracted from image!")
            st.write(image_text)
            st.session_state.history.append(("Image Upload", image_text))

# Display History on the Left Side (Sidebar)
st.sidebar.subheader("Interaction History")
if st.session_state.history:
    for i, (user_input, response_output) in enumerate(st.session_state.history):
        st.sidebar.write(f"**Interaction {i + 1}:**")
        st.sidebar.write(f"**User Input:** {user_input}")
        st.sidebar.write(f"**Response Output:** {response_output}")
else:
    st.sidebar.write("No history yet.")

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
        st.session_state.history.append(("Translation", translated_text))
