import gradio as gr
import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer, pipeline, AutoModelForCausalLM

# Your Hugging Face token
HF_TOKEN = "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"  # Replace with your token

# Load Models and Pipelines
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    return summarizer

# Load translation model (using MarianMT for Chinese-to-English translation)
def load_translation_model():
    model_name = "Helsinki-NLP/opus-mt-zh-en"  # Chinese to English translation model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    translator = pipeline("translation", model=model, tokenizer=tokenizer)
    return translator

def load_code_generation_model():
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Use the appropriate Llama2 model for code generation
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)  # Correct model class for Llama2
    code_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)  # Use text-generation pipeline
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

# Gradio Interface Functions

def summarize_pdf(file):
    pdf_text = extract_text_from_pdf(file)
    if pdf_text:
        summary = summarize_text(pdf_text)
        return summary
    else:
        return "Failed to extract text. Please check your PDF file."

def summarize_text_input(text):
    return summarize_text(text)

def translate_text_input(text):
    translation = translator(text)
    return translation[0]['translation_text']

def generate_code_input(prompt):
    generated_code = code_generator(prompt, max_length=200)
    return generated_code[0]['generated_text']

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Interactive Chat Application with Text and PDF Summarization")
    gr.Markdown("Summarize content from uploaded PDFs, translate, or generate code.")
    
    with gr.Tab("Upload PDF"):
        pdf_input = gr.File(file_types=["pdf"], label="Upload PDF")
        pdf_output = gr.Textbox(label="Summarized PDF Content")
        pdf_input.upload(summarize_pdf, pdf_input, pdf_output)

    with gr.Tab("Enter Text Manually"):
        manual_text_input = gr.Textbox(label="Enter text to summarize", lines=5)
        manual_text_output = gr.Textbox(label="Summarized Text")
        manual_text_input.submit(summarize_text_input, manual_text_input, manual_text_output)
    
    with gr.Tab("Translate Text"):
        translate_input = gr.Textbox(label="Enter text to translate", lines=5)
        translate_output = gr.Textbox(label="Translated Text")
        translate_input.submit(translate_text_input, translate_input, translate_output)

    with gr.Tab("Generate Code"):
        code_input = gr.Textbox(label="Enter prompt for code generation", lines=5)
        code_output = gr.Textbox(label="Generated Code")
        code_input.submit(generate_code_input, code_input, code_output)

demo.launch(share=True)
