import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
"""
BlipProcessor, BlipForConditionalGeneration:
These are from the Hugging Face transformers library. 

BlipProcessor handles input preprocessing for the BLIP model, 
BlipForConditionalGeneration is the BLIP model itself, which generates image descriptions.

T5Tokenizer, T5ForConditionalGeneration: These are also from transformers. 
T5Tokenizer is used to convert text into tokens and back, 
and T5ForConditionalGeneration is the T5 model for text generation tasks like summarization

"""
import sentencepiece as spm 
from PIL import Image

# Load the BLIP model and processor for image description
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Loading the models

#This processor handles the conversion of images to tensors that can be fed into the BLIP model.
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

#image-captioning model itself
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Load the T5 model for text summarization
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

def generate_description(image: Image.Image) -> str:
    """
    Generates a detailed description for the given image.

    Parameters:
    image (PIL.Image.Image): The input image.

    Returns:
    str: The generated description.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = blip_model.generate(**inputs)
    description = processor.decode(outputs[0], skip_special_tokens=True)
    return description

def summarize_text_and_image(description: str, ocr_text: str) -> str:
    """
    Generates a summary combining the image description and OCR-extracted text.

    Parameters:
    description (str): The generated description of the image.
    ocr_text (str): The text extracted from the image using OCR.

    Returns:
    str: The generated summary.
    """
    combined_input = f"Image Description: {description} Text: {ocr_text}"
    input_ids = t5_tokenizer.encode(combined_input, return_tensors="pt", truncation=True).to(device)
    outputs = t5_model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
    summary = t5_tokenizer.decode(outputs[0], skip_special_tokens=True) 
    return summary
