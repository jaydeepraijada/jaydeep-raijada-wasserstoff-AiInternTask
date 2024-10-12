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
from .database_handler import DatabaseHandler
from multiprocessing import Pool

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

def process_pdf_mp(pdf_file, config):
    # This function will run in a separate process
    text = extract_text_from_pdf(pdf_file)
    summary = summarize_text(text, config)
    keywords = extract_keywords(text)
    return pdf_file, text, summary, keywords

async def process_pdf(pdf_file, config, db_handler, performance_monitor):
    start_time = time.time()
    try:
        print(f"Starting to process {pdf_file}")
        
        # Record PDF size
        pdf_size = os.path.getsize(pdf_file)
        performance_monitor.record_pdf_size(pdf_size)
        
        # Use a process pool to handle CPU-intensive tasks
        with Pool() as pool:
            text_extraction_start = time.time()
            pdf_file, text, summary, keywords = await asyncio.to_thread(
                pool.apply, process_pdf_mp, (pdf_file, config)
            )
            text_extraction_time = time.time() - text_extraction_start
            performance_monitor.record_task_timing('text_extraction', text_extraction_time)
        
        # Database update (I/O-bound, keep it async)
        db_update_start = time.time()
        await db_handler.update_document(pdf_file, summary, keywords)
        db_update_time = time.time() - db_update_start
        performance_monitor.record_task_timing('db_operation', db_update_time)
        
        processing_time = time.time() - start_time
        print(f"Processed {pdf_file} in {processing_time:.2f} seconds")
        
        return {
            'pdf_file': pdf_file,
            'summary': summary,
            'keywords': keywords,
            'processing_time': processing_time,
            'db_update_time': db_update_time
        }
    except Exception as e:
        print(f"Error processing {pdf_file}: {str(e)}")
        performance_monitor.record_error(type(e).__name__)
        return {
            'pdf_file': pdf_file,
            'summary': "",
            'keywords': [],
            'processing_time': time.time() - start_time,
            'error': str(e)
        }

async def process_pdfs_in_folder(folder_path, config, db_handler, semaphore, performance_monitor):
    """
    Process all PDF files in a given folder concurrently.
    
    Args:
        folder_path (str): Path to the folder containing PDF files.
        config (Config): Configuration object.
        db_handler (DatabaseHandler): Database handler object.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent tasks.
        performance_monitor (PerformanceMonitor): Performance monitoring object.
    
    Returns:
        list: List of processing results for each PDF.
    """
    pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".pdf")]
    
    async def process_with_semaphore(pdf_file):
        async with semaphore:
            performance_monitor.task_started()
            try:
                return await process_pdf(pdf_file, config, db_handler, performance_monitor)
            finally:
                performance_monitor.task_ended()
    
    results = await asyncio.gather(*[process_with_semaphore(pdf_file) for pdf_file in pdf_files])
    return results
