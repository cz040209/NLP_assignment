import streamlit as st
import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer, AutoModelForQuestionAnswering
import torch

# Your Hugging Face token
HF_TOKEN = "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"  # Replace with your token

# Load Summarization Model and Tokenizer
@st.cache_resource
def load_summarization_model():
    model_name = "facebook/bart-large-cnn"  # Replace with your model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    return tokenizer, model

# Load MarianMT Model and Tokenizer for Translation (Chinese-English)
@st.cache_resource
def load_translation_model(model_name):
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Load Question Answering Model and Tokenizer
@st.cache_resource
def load_qa_model():
    model_name = "distilbert-base-uncased-distilled-squad"  # Question Answering model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    return tokenizer, model

# Initialize models and tokenizers
summarization_tokenizer, summarization_model = load_summarization_model()
qa_tokenizer, qa_model = load_qa_model()

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

# Function to answer a question based on context text
def answer_question(question, context):
    inputs = qa_tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True, padding=True)
    answer_start_scores, answer_end_scores = qa_model(**inputs)

    # Get the most likely start and end positions of the answer
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores)

    # Convert tokens to the corresponding text
    answer = qa_tokenizer.convert_tokens_to_string(
        qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end + 1])
    )

    return answer

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

# Question Answering Section
st.subheader("Ask a Question")

# Ask a question if there is context text
if context_text:
    question = st.text_input("Enter your question:")
    if question.strip():
        if st.button("Get Answer"):
            with st.spinner("Getting answer..."):
                answer = answer_question(question, context_text)
            st.success("Answer generated!")
            st.write(answer)

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
