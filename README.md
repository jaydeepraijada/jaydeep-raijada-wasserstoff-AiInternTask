# PDF Processing and Summarization Tool

## Overview

This project is an asynchronous PDF processing tool that extracts text from PDF files, generates summaries, extracts keywords, and stores the results in a MongoDB database. It's designed to handle multiple PDFs concurrently, making it efficient for processing large numbers of documents.

## Features

- Asynchronous PDF text extraction using PyMuPDF
- Fallback to OCR using EasyOCR for scanned documents
- Text summarization using a pre-trained PEGASUS model
- Keyword extraction using YAKE
- Concurrent processing of multiple PDFs
- MongoDB integration for storing results
- Performance monitoring and reporting

## Requirements

- Python 3.7+
- MongoDB
- CUDA-capable GPU (optional, for faster processing)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/pdf-processing-tool.git
   cd pdf-processing-tool
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your MongoDB connection by creating a `.env` file with your MongoDB URI:
   ```
   MONGODB_URI=mongodb://localhost:27017/
   ```

## Usage

1. Place your PDF files in the designated folder (default is `C:\Users\acer\Desktop\Task_pdfs`).

2. Run the main script:
   ```
   python main.py
   ```

3. The script will process all PDFs in the folder, generating summaries and extracting keywords.

4. Results will be stored in the MongoDB database and a performance report will be generated.

## Configuration

You can modify the following parameters in `main.py`:

- `PDF_FOLDER_PATH`: Path to the folder containing PDF files
- `MONGODB_URI`: MongoDB connection string
- `DATABASE_NAME`: Name of the MongoDB database
- `COLLECTION_NAME`: Name of the MongoDB collection
- `MODEL_NAME`: Name of the pre-trained model for summarization
- `MAX_LENGTH`, `MIN_SUMMARY_LENGTH`, `MAX_SUMMARY_LENGTH`: Parameters for summary generation

## Performance Monitoring

The tool includes a `PerformanceMonitor` class that tracks various metrics during processing:

- Total execution time
- Number of PDFs processed (successful and failed)
- Processing times (total, average, min, max, median)
- CPU and memory usage
- Concurrency metrics

A performance report is generated after processing and saved as `performance_report.txt`.

## Testing

Run the unit tests using:

python -m unittest discover tests
