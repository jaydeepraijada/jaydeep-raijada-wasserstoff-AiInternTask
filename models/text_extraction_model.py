import easyocr
import numpy as np

# Load the EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

def extract_text(image): 
    # Convert the image to an array for OCR
    image_np = np.array(image)
    
    # Perform OCR on the image
    result = reader.readtext(image_np)
    
    # Extract text from the OCR result
    extracted_text = []
    for (bbox, text, prob) in result:
        extracted_text.append(text)
    
    return extracted_text
