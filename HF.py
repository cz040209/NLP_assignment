import streamlit as st
import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
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

# Load Question-Answering Model and Tokenizer
@st.cache_resource
def load_qa_model():
    model_name = "deepset/roberta-base-squad2"  # Replace with a Q&A model
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

# Function to answer questions based on a context
def answer_question(context, question):
    inputs = qa_tokenizer(question, context, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = qa_model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)  # Start of the answer
        answer_end = torch.argmax(outputs.end_logits) + 1  # End of the answer
    answer = qa_tokenizer.decode(inputs.input_ids[0][answer_start:answer_end], skip_special_tokens=True)
    return answer if answer.strip() else "No answer found."

# Streamlit App
st.title("Interactive Summarization and Q&A Application")
st.subheader("Summarize content from PDFs or manual input and ask questions about it.")

# Option to choose between PDF upload or manual input
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

# Q&A Section
if context_text:
    st.subheader("Ask a Question About the Text")
    question = st.text_input("Enter your question:", "")
    if st.button("Get Answer"):
        if question.strip():
            with st.spinner("Finding the answer..."):
                answer = answer_question(context_text, question)
            st.success("Answer:")
            st.write(answer)
        else:
            st.info("Please enter a valid question.")
