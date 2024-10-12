import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = None
model = None

def load_model(model_name):
    global tokenizer, model
    try:
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        print("Tokenizer and model loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer or model: {str(e)}")
        raise

# Load the model once at the start
load_model("nsi319/legal-pegasus")

async def summarize_text(extracted_text, config):
    global tokenizer, model
    """
    Generates a summary of the given text using a pre-trained model.
    
    Args:
        extracted_text (str): The text to be summarized.
        config (Config): Configuration object containing summarization parameters.
    
    Returns:
        str: The generated summary.
    """
    
    if len(extracted_text) < 1000:  # Short document
        max_length, min_length = 100, 30
    elif len(extracted_text) < 10000:  # Medium document
        max_length, min_length = 200, 50
    else:  # Long document
        max_length, min_length = 300, 100

    try:
        input_tokenized = tokenizer(
            extracted_text,
            return_tensors='pt',
            max_length=config.max_length,
            truncation=True,
        ).to(device)

        summary_ids = model.generate(
            input_tokenized['input_ids'],
            num_beams=2,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            min_length=min_length,
            max_length=max_length,
            early_stopping=True
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}")
        return ""
