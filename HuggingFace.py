import os
import streamlit as st
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForSeq2SeqLM
from PIL import Image
import pdfplumber
import torch
import speech_recognition as sr  # For audio-to-text functionality
from gtts import gTTS

# Your Hugging Face token
HF_TOKEN = "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"  # Replace with your token

# Local path where you've downloaded and unzipped the model
LLAMA3_PATH = r"C:\Users\chenz\Downloads\llama3-main.zip\llama3-main"

# Function to load Llama3 model from local directory
@st.cache_resource
def load_llama3_model():
    model_name = LLAMA3_PATH  # Path to the local model

    # Load tokenizer and model from the local path
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # Using AutoTokenizer for simplicity
    model = LlamaForCausalLM.from_pretrained(model_name)  # Load Llama model locally

    return tokenizer, model

# Load Summarization Model and Tokenizer
@st.cache_resource
def load_summarization_model(model_choice="Llama3"):
    if model_choice == "Llama3":
        tokenizer, model = load_llama3_model()  # Load the local Llama3 model
    else:
        raise ValueError(f"Unsupported model choice: {model_choice}")

    return tokenizer, model

# Initialize models and tokenizers based on user selection
model_choice = st.selectbox("Select Model for Summarization:", ("Llama3",))

# Ensure a model is chosen before proceeding
if model_choice not in ["Llama3"]:
    st.warning("Please select a valid model for summarization and conversation.")

summarization_tokenizer, summarization_model = load_summarization_model(model_choice)

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

# Function for Image to Text (BLIP)
def image_to_text(image_file):
    # Assuming BLIP model is used for image-to-text
    processor, model = load_blip_model()
    image = Image.open(image_file).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description

# History storage - will store interactions as tuples (user_input, response_output)
if 'history' not in st.session_state:
    st.session_state.history = []

# Custom CSS for a more premium look
st.markdown("""
    <style>
        .css-1d391kg {
            background-color: #1c1f24;  /* Dark background */
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .css-1v0m2ju {
            background-color: #282c34;  /* Slightly lighter background */
        }
        .css-13ya6yb {
            background-color: #61dafb;  /* Button color */
            border-radius: 5px;
            padding: 10px 20px;
            color: white;
            font-size: 16px;
            font-weight: bold;
        }
        .css-10trblm {
            font-size: 18px;
            font-weight: bold;
            color: #282c34;
        }
        .css-3t9iqy {
            color: #61dafb;
            font-size: 20px;
        }
        .botify-title {
            font-family: 'Arial', sans-serif;
            font-size: 48px;
            font-weight: bold;
            color: #61dafb;
            text-align: center;
            margin-top: 50px;
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Botify Title
st.markdown('<h1 class="botify-title">Botify</h1>', unsafe_allow_html=True)

# Option to choose between PDF upload, manual input, or translation
option = st.selectbox("Choose input method:", ("Upload PDF", "Enter Text Manually", "Upload Audio", "Upload Image"))

context_text = ""

# Handling different options
if option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text:
            st.success("Text extracted successfully!")
            st.text_area("Extracted Text (Preview)", pdf_text[:2000], height=200)
            context_text = pdf_text

            # Summarize text
            st.subheader("Summarize the PDF Content")
            if st.button("Summarize PDF", use_container_width=True):
                with st.spinner("Summarizing text..."):
                    summary = summarize_text(pdf_text)
                st.success("Summary generated!")
                st.write(summary)
                st.session_state.history.append(("PDF Upload", summary))
        else:
            st.error("Failed to extract text. Please check your PDF file.")

elif option == "Enter Text Manually":
    manual_text = st.text_area("Enter your text below:", height=200)
    if manual_text.strip():
        context_text = manual_text

        st.subheader("Summarize the Entered Text")
        if st.button("Summarize Text", use_container_width=True):
            with st.spinner("Summarizing text..."):
                summary = summarize_text(manual_text)
            st.success("Summary generated!")
            st.write(summary)
            st.session_state.history.append(("Manual Text", summary))
    else:
        st.info("Please enter some text to summarize.")

elif option == "Upload Audio":
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"], label_visibility="collapsed")

    if audio_file:
        with st.spinner("Transcribing audio to text..."):
            try:
                transcription = audio_to_text(audio_file)
                st.success("Transcription successful!")
                st.write(transcription)
                st.session_state.history.append(("Audio Upload", transcription))
            except Exception as e:
                st.error(f"Error: {e}")

elif option == "Upload Image":
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if image_file:
        with st.spinner("Extracting text from image..."):
            image_text = image_to_text(image_file)
            st.success("Text extracted from image!")
            st.write(image_text)
            st.session_state.history.append(("Image Upload", image_text))

# Sidebar for Interaction History with improved layout
st.sidebar.subheader("Interaction History")
if st.session_state.history:
    for i, (user_input, response_output) in enumerate(st.session_state.history):
        st.sidebar.write(f"**Interaction {i + 1}:**")
        st.sidebar.write(f"**User Input:** {user_input}")
        st.sidebar.write(f"**Response Output:** {response_output}")
else:
    st.sidebar.write("No history yet.")
