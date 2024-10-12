import os
import asyncio
import time
import logging
import fitz  # PyMuPDF
import easyocr
import numpy as np
from PIL import Image
from .text_summarizer import summarize_text
from .keyword_extractor import extract_keywords
from .database_handler import update_mongodb

logger = logging.getLogger(__name__)

async def extract_text_from_pdf(pdf_path):
    """
    Extracts text content from a PDF file using PyMuPDF.
    Falls back to EasyOCR if PyMuPDF fails to extract text.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text from the PDF.
    """
    try:
        # Attempt to extract text using PyMuPDF
        document = fitz.open(pdf_path)
        extracted_text = ""
        for page in document:
            extracted_text += page.get_text()
        document.close()
        
        # If PyMuPDF extracted no text, fall back to OCR
        if not extracted_text.strip():
            return await ocr_pdf(pdf_path)
        
        return extracted_text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

async def ocr_pdf(pdf_path):
    """
    Performs OCR on a PDF file using EasyOCR.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text from the PDF using OCR.
    """
    try:
        reader = easyocr.Reader(['en'])  # Initialize EasyOCR with English language
        document = fitz.open(pdf_path)
        extracted_text = ""
        
        for page in document:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            results = reader.readtext(np.array(img))
            page_text = " ".join([result[1] for result in results])
            extracted_text += page_text + "\n\n"
        
        document.close()
        return extracted_text
    except Exception as e:
        logger.error(f"Error performing OCR on {pdf_path}: {str(e)}")
        return ""

async def worker(pdf_file, config):
    """
    Processes a single PDF file: extracts text, generates summary, extracts keywords,
    and updates the database.
    
    Args:
        pdf_file (str): Path to the PDF file.
        config (dict): Configuration dictionary containing necessary parameters.
    
    Returns:
        tuple: (pdf_file, summary, keywords, processing_time)
    """
    start_time = time.time()
    try:
        logger.info(f"Starting to process {pdf_file}")
        
        text_extraction_start = time.time()
        text = await extract_text_from_pdf(pdf_file)
        logger.info(f"Text extraction for {pdf_file} took {time.time() - text_extraction_start:.2f} seconds")
        
        summary_start = time.time()
        summary = await summarize_text(text)
        logger.info(f"Summarization for {pdf_file} took {time.time() - summary_start:.2f} seconds")
        
        keyword_start = time.time()
        keywords = extract_keywords(text)
        logger.info(f"Keyword extraction for {pdf_file} took {time.time() - keyword_start:.2f} seconds")
        
        db_update_start = time.time()
        await update_mongodb(pdf_file, summary, keywords)
        logger.info(f"Database update for {pdf_file} took {time.time() - db_update_start:.2f} seconds")
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {pdf_file} in {processing_time:.2f} seconds")
        return pdf_file, summary, keywords, processing_time
    except Exception as e:
        logger.error(f"Error processing {pdf_file}: {str(e)}")
        return pdf_file, "", [], time.time() - start_time

async def process_pdfs_in_folder(folder_path, config, semaphore, performance_monitor):
    pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".pdf")]
    
    async def process_with_semaphore(pdf_file):
        async with semaphore:
            performance_monitor.task_started()
            try:
                return await worker(pdf_file, config)
            finally:
                performance_monitor.task_ended()
    
    results = await asyncio.gather(*[process_with_semaphore(pdf_file) for pdf_file in pdf_files])
    return results
