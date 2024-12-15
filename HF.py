import os
import streamlit as st
import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GPT2LMHeadModel, GPT2Tokenizer, BlipProcessor, BlipForConditionalGeneration
import torch
import speech_recognition as sr  # For audio-to-text functionality
from PIL import Image

# Your Hugging Face token
HF_TOKEN = "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"  # Replace with your token

# Set up the BLIP model for image-to-text
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Load Summarization Model and Tokenizer
@st.cache_resource
def load_summarization_model(model_choice="BART"):
    if model_choice == "BART":
        model_name = "facebook/bart-large-cnn"  # BART model for summarization
    elif model_choice == "Gemini":
        model_name = "google/flan-t5-large"  # Example Gemini model (use correct model name for Gemini)
    else:
        model_name = "gpt2"  # GPT-2 model for comparison
    
    if model_choice == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=HF_TOKEN)
    
    return tokenizer, model

# Initialize models and tokenizers
model_choice = st.selectbox("Select Model for Summarization:", ("BART", "Gemini", "GPT-2"))

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

# Translation Section with clean layout
st.subheader("Translate Text")

# Choose translation direction (English ↔ Chinese)
target_language = st.selectbox("Choose translation direction:", ("English to Chinese", "Chinese to English"))

if context_text:
    st.subheader("Translate the Text")
    if st.button("Translate Text", use_container_width=True):
        with st.spinner("Translating text..."):
            # Load translation model
            translation_model, translation_tokenizer = load_summarization_model(model_choice=model_choice)  # Can select Gemini for translation
            # Prepare the text for translation
            inputs = translation_tokenizer(context_text, return_tensors="pt", padding=True)
            
            # Generate translation
            translated = translation_model.generate(**inputs)
            translated_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)

        st.success(f"Translated text ({target_language}):")
        st.write(translated_text)
        st.session_state.history.append(("Translation", translated_text))

# Add a Conversation AI section
st.subheader("Chat with Botify")

# User input for chat
user_query = st.text_input("Enter your query:", key="chat_input", placeholder="Type something to chat!")

# Process the query if entered
if user_query:
    with st.spinner("Generating response..."):
        # Load model based on selection
        if model_choice == "GPT-2":
            conversational_model = GPT2LMHeadModel.from_pretrained("gpt2")
            conversational_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        elif model_choice == "BART":
            conversational_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")
            conversational_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        elif model_choice == "Gemini":
            conversational_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
            conversational_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

        # Generate the response
        inputs = conversational_tokenizer(user_query, return_tensors="pt")
        response = conversational_model.generate(inputs["input_ids"], max_length=200)
        bot_reply = conversational_tokenizer.decode(response[0], skip_special_tokens=True)

    st.write(f"Botify: {bot_reply}")
    st.session_state.history.append(("User Query", bot_reply))
