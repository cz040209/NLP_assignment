import streamlit as st
import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Your Hugging Face token
HF_TOKEN = "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"  # Replace with your token

# Load Models and Pipelines
@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    return summarizer

def load_translation_model():
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with your translation model, if needed
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    translator = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)  # Example: English to French translation
    return translator

def load_code_generation_model():
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Use the appropriate Llama2 model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    code_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return code_generator

# Initialize the summarizer, translator, and code generator pipelines
summarizer = load_model("facebook/bart-large-cnn")
translator = load_translation_model()
code_generator = load_code_generation_model()

# Function to split text into manageable chunks
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
        result = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
        if result:
            summaries.append(result[0]['summary_text'])
    final_summary = " ".join(summaries)
    return final_summary if summaries else "No summary available."

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return " ".join(page.extract_text() for page in pdf.pages if page.extract_text())

# Streamlit App
st.title("Interactive Chat Application with Text and PDF Summarization")
st.subheader("Summarize content from uploaded PDFs, translate, or generate code.")

# Option to choose between PDF upload, manual input, translation, or code generation
option = st.radio("Choose input method:", ("Upload PDF", "Enter Text Manually", "Translate Text", "Generate Code"))

if option == "Upload PDF":
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text:
            st.success("Text extracted successfully!")
            st.text_area("Extracted Text (Preview)", pdf_text[:2000], height=200)
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
        st.subheader("Summarize the Entered Text")
        if st.button("Summarize Text"):
            with st.spinner("Summarizing text..."):
                summary = summarize_text(manual_text)
            st.success("Summary generated!")
            st.write(summary)
    else:
        st.info("Please enter some text to summarize.")

elif option == "Translate Text":
    st.subheader("Translate Text")
    input_text = st.text_area("Enter text to translate:", height=200)
    if input_text.strip():
        target_language = st.selectbox("Select target language:", ("French", "Spanish", "German", "Italian"))
        if st.button("Translate"):
            with st.spinner("Translating text..."):
                translation = translator(input_text)
            st.success("Translation generated!")
            st.write(translation[0]['translation_text'])
    else:
        st.info("Please enter some text to translate.")

elif option == "Generate Code":
    st.subheader("Generate Code")
    prompt = st.text_area("Enter prompt for code generation:", height=200)
    if prompt.strip():
        if st.button("Generate Code"):
            with st.spinner("Generating code..."):
                generated_code = code_generator(prompt)
            st.success("Code generated!")
            st.code(generated_code[0]['generated_text'], language="python")
    else:
        st.info("Please enter a prompt to generate code.")

