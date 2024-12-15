import os
import streamlit as st
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import pdfplumber
import speech_recognition as sr  # For audio-to-text functionality
from PIL import Image
import pytesseract  # For OCR (Image to Text)

# Set the path to the tesseract executable (required for image-to-text conversion)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\chenz\Downloads\tesseract-ocr-w64-setup-5.5.0.20241111\tesseract.exe'

# Your Hugging Face token for Llama2
HF_TOKEN = "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"  # Replace with your token

# Load Llama 2 Model and Tokenizer for Conversational AI (Chatbot)
@st.cache_resource
def load_llama2_conversational_model():
    model_name = "meta-llama/Llama-2-7b-hf"  # Replace with the appropriate Llama 2 model name
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Initialize the conversational model and tokenizer
conversational_tokenizer, conversational_model = load_llama2_conversational_model()

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

# Function to generate conversational responses (Chatbot)
def generate_conversation_response(user_input, context_text=""):
    # Prepare the input to the model
    prompt = f"Human: {user_input}\nAI:"

    # Concatenate the context (if any)
    if context_text:
        prompt = f"{context_text}\n{prompt}"

    # Tokenize the input prompt
    inputs = conversational_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    outputs = conversational_model.generate(
        inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True
    )
    
    response = conversational_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

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

# Function to summarize extracted text
def summarize_text(text, max_tokens=1024):
    # Split the text into chunks to avoid exceeding token limit for Llama2
    chunks = split_text(text, max_tokens=max_tokens)
    
    summary = ""
    for chunk in chunks:
        # Add each chunk to the prompt for summarization
        prompt = f"Summarize the following text:\n\n{chunk}"
        
        # Tokenize and generate the summary
        inputs = conversational_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        summary_output = conversational_model.generate(
            inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True
        )
        
        # Decode the output and append to the summary
        summary += conversational_tokenizer.decode(summary_output[0], skip_special_tokens=True)
        summary += "\n"

    return summary

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

elif option == "Enter Text Manually":
    manual_text = st.text_area("Enter your text below:", height=200)
    if manual_text.strip():
        context_text = manual_text

elif option == "Upload Audio":
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"], label_visibility="collapsed")

    if audio_file:
        with st.spinner("Transcribing audio to text..."):
            try:
                transcription = audio_to_text(audio_file)
                st.success("Transcription successful!")
                st.write(transcription)
                context_text = transcription
            except Exception as e:
                st.error(f"Error: {e}")

elif option == "Upload Image":
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if image_file:
        with st.spinner("Extracting text from image..."):
            image_text = image_to_text(image_file)
            st.success("Text extracted from image!")
            st.write(image_text)
            context_text = image_text

# Summarize extracted text if available
if context_text:
    summarize_button = st.button("Summarize Extracted Text")
    
    if summarize_button:
        with st.spinner("Summarizing text..."):
            summary = summarize_text(context_text)
        st.success("Summary generated!")
        st.text_area("Summary", summary, height=200)

# Handling the conversational part
st.subheader("Ask the AI a Question")

if context_text:
    user_input = st.text_input("Your Question", "")

    if user_input:
        response = generate_conversation_response(user_input, context_text)
        st.write(f"AI: {response}")
        st.session_state.history.append(("User Question", response))

# Sidebar for Interaction History with improved layout
st.sidebar.subheader("Interaction History")
if st.session_state.history:
    for i, (user_input, response_output) in enumerate(st.session_state.history):
        st.sidebar.write(f"**Interaction {i + 1}:**")
        st.sidebar.write(f"**User Input:** {user_input}")
        st.sidebar.write(f"**Response Output:** {response_output}")
else:
    st.sidebar.write("No history yet.")
