import os
import streamlit as st
import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
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
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=HF_TOKEN)
    return tokenizer, model

# Load Llama 2 Model for Q&A
@st.cache_resource
def load_llama2_model():
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with your preferred Llama 2 variant
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN)
    return tokenizer, model

# Initialize models and tokenizers
summarization_tokenizer, summarization_model = load_summarization_model()
llama2_tokenizer, llama2_model = load_llama2_model()

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

# Function for Q&A using Llama 2
def chat_with_llama2(query):
    inputs = llama2_tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = llama2_model.generate(
        inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True
    )
    response = llama2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

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

# Chat with Botify (Llama 2)
st.subheader("Chat with Botify")

# User input for chat
user_query = st.text_input("Ask something:", key="chat_input", placeholder="Type your query here...")

# Process the query if entered
if user_query:
    with st.spinner("Botify is thinking..."):
        try:
            bot_response = chat_with_llama2(user_query)
            st.markdown(f"**Botify:** {bot_response}")
            st.session_state.history.append((user_query, bot_response))
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Sidebar for Interaction History
st.sidebar.subheader("Interaction History")
if st.session_state.history:
    for i, (user_input, response_output) in enumerate(st.session_state.history):
        st.sidebar.write(f"**Interaction {i + 1}:**")
        st.sidebar.write(f"**User Input:** {user_input}")
        st.sidebar.write(f"**Response Output:** {response_output}")
else:
    st.sidebar.write("No history yet.")
