import os
import streamlit as st
import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from gtts import gTTS
from PIL import Image
import speech_recognition as sr  # For audio-to-text functionality

# Load environment variables from .env file (optional, for local development)
from dotenv import load_dotenv
load_dotenv()  # Ensure that this file contains your API keys

# Your Hugging Face API token (Replace with your Hugging Face token here)
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "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")  # Replace with your Hugging Face token

# Your Azure API key (make sure it's stored in environment variable or directly here)
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "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")  # Replace with your Azure API key
AZURE_ENDPOINT = "https://models.inference.ai.azure.com"  # Replace with your Azure endpoint

# Set up the BLIP model for image-to-text
def load_blip_model():
    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Function for Text-to-Speech (Text to Audio)
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')  # You can change language if needed
    tts.save("response.mp3")  # Save the speech as an audio file
    # Provide a link to download or play the audio
    st.audio("response.mp3", format="audio/mp3")
    os.remove("response.mp3")  # Clean up the temporary audio file

# Function to load the Azure Llama model for summarization or chat
@st.cache_resource
def load_llama_model():
    client = ChatCompletionsClient(
        endpoint=AZURE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_API_KEY),
    )
    return client

# Function to summarize text using Llama (Azure)
def summarize_with_llama(text):
    client = load_llama_model()
    response = client.complete(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ],
        model="Meta-Llama-3.1-70B-Instruct",  # Azure Llama model
        temperature=0.7,
        max_tokens=150,
    )
    return response.choices[0].message.content

# Load Summarization Model and Tokenizer for Hugging Face models (BART, T5, Llama)
@st.cache_resource
def load_summarization_model(model_choice="BART"):
    if model_choice == "BART":
        model_name = "facebook/bart-large-cnn"  # BART model for summarization
    elif model_choice == "T5":
        model_name = "t5-large"  # T5 model for summarization
    elif model_choice == "Llama3":
        model_name = "meta-llama/Llama-3.3-70B-Instruct"  # Llama 3 model for summarization
    else:
        raise ValueError(f"Unsupported model choice: {model_choice}")
    
    # Ensure consistent use of the selected model for both summarization and conversation
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGINGFACE_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=HUGGINGFACE_TOKEN)

    return tokenizer, model

# Initialize models and tokenizers based on user selection
model_choice = st.selectbox("Select Model for Summarization:", ("BART", "T5", "Llama3"))

# Ensure a model is chosen before proceeding
if model_choice not in ["BART", "T5", "Llama3"]:
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
        .css-1d391kg { background-color: #1c1f24; color: white; font-family: 'Arial', sans-serif; } 
        .css-1v0m2ju { background-color: #282c34; } 
        .css-13ya6yb { background-color: #61dafb; border-radius: 5px; padding: 10px 20px; color: white; font-size: 16px; font-weight: bold; } 
        .css-10trblm { font-size: 18px; font-weight: bold; color: #282c34; } 
        .css-3t9iqy { color: #61dafb; font-size: 20px; } 
        .botify-title { font-family: 'Arial', sans-serif; font-size: 48px; font-weight: bold; color: #61dafb; text-align: center; margin-top: 50px; margin-bottom: 30px; } 
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
            st.session_state.history.append(("Manual Input", summary))

elif option == "Upload Audio":
    uploaded_audio = st.file_uploader("Upload Audio", type=["wav", "mp3", "flac"], label_visibility="collapsed")

    if uploaded_audio:
        with st.spinner("Converting audio to text..."):
            audio_text = audio_to_text(uploaded_audio)
        st.success("Audio converted to text!")
        st.text_area("Converted Audio Text", audio_text[:2000], height=200)
        context_text = audio_text

        # Summarize the audio content
        st.subheader("Summarize Audio Content")
        if st.button("Summarize Audio", use_container_width=True):
            with st.spinner("Summarizing text..."):
                summary = summarize_text(audio_text)
            st.success("Summary generated!")
            st.write(summary)
            st.session_state.history.append(("Audio Upload", summary))

elif option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_image:
        with st.spinner("Extracting description from image..."):
            image_description = image_to_text(uploaded_image)
        st.success("Description generated from image!")
        st.write(image_description)
        context_text = image_description

        # Summarize the image description
        st.subheader("Summarize Image Description")
        if st.button("Summarize Image Description", use_container_width=True):
            with st.spinner("Summarizing description..."):
                summary = summarize_text(image_description)
            st.success("Summary generated!")
            st.write(summary)
            st.session_state.history.append(("Image Upload", summary))

# Display chat history
if st.session_state.history:
    st.subheader("Conversation History")
    for idx, (user_input, bot_output) in enumerate(st.session_state.history):
        st.write(f"**{user_input}**: {bot_output}")
