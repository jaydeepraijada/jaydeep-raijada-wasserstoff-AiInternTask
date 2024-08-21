
# AI Pipeline for Image Segmentation and Object Analysis

This project is an AI-driven pipeline designed to process images for segmentation, object extraction, identification, text/data extraction, and summarization. The pipeline is implemented using various deep learning models and tools, and provides an interactive user interface via Streamlit. The project is modular, with clear separation of concerns


## Features

- Image Segmentation: Segments the input image into individual objects.
- Object Extraction: Extracts and stores segmented objects with unique IDs and metadata.
- Object Identification: Identifies objects within the extracted images using YOLOv8.
- Text/Data Extraction: Extracts text or data from objects using EasyOCR.
- Multimodal Summarization: Summarizes the attributes of the objects and the extracted text using BLIP and T5.
- User Interface: Provides a Streamlit-based UI for user interaction with the pipeline

## Usage

Running the Streamlit Application
To launch the Streamlit UI and interact with the pipeline:

bash
streamlit run streamlit_app/app.py

Example Workflow

- Upload an Image: The user uploads an image via the Streamlit UI.
- Segmentation: The image is segmented, and the objects are extracted and saved.
- Object Identification: Each object is identified, and its data is processed.
- Text Extraction: Text is extracted from each object using OCR.
- Summarization: The attributes of each object and the extracted text are summarized.
- Data Mapping and Output Generation: The final output image is annotated, and a table containing all mapped data is generated.

--For now because of some system related issues, I haven't been able to complete the data mapping and output generation step, will be completeing that soon and commiting the changes.
