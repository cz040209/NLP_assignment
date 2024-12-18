from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

# The direct URL to the model files (from Meta's official Llama repo)
MODEL_URL = "https://llama3-3.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoibW1lMm5xc2NraHp4NzV3eHkwbW5ncjJ3IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTMubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczNDcwNDMwNH19fV19&Signature=Hkh%7EU384aYXeUI8KSN0cehcrNQpSlAWv3eygk-8qL3NHJ-zMEtzJtpYk5KtJRJvt5Qa-cfWvUmxMjJPK2JfaRQwDvPId4ka2fOwc7f7lOIsxKKfjZ0kbeXYHwbmYImVyxZdrAvQGnC0RrCtxNEuFDQ%7Es5%7EwberM2vXNezaKxTZMtzom6q5Y0w2PnXJDwK5rTAsglBd4PTITdnUS1f3vGBToZiPZfwxuUAdpHeEAlHDVlVM9wsZS%7Ewb-Sxonri0Z3MspzQx7Bd1VXS3dK%7E1M85EBRCljVNjdwiFNwnd7qdW7-QpQ31SfAtMiVfaqJb19QS18E9XNJorfppPMvuqPuyw__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1054067176517874"

# Load the Llama model and tokenizer directly from the URL
def load_llama3_model_from_url():
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_URL)  # No Hugging Face token needed if it's public
    model = LlamaForCausalLM.from_pretrained(MODEL_URL)  # Similarly for the model
    return tokenizer, model

# Example: Load model and tokenizer
summarization_tokenizer, summarization_model = load_llama3_model_from_url()

# Function to summarize text using the Llama model
def summarize_text(text):
    inputs = summarization_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    summary_ids = summarization_model.generate(
        inputs["input_ids"], max_length=150, min_length=40, num_beams=4, early_stopping=True
    )
    summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example text to summarize
text = "This is an example text that we want to summarize using Llama model."
summary = summarize_text(text)
print("Summary:", summary)
