import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self):
        self.pdf_folder_path = r"C:\Users\acer\Desktop\Task_pdfs"
        self.mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        self.database_name = 'pdf_database1'
        self.collection_name = 'pdf_documents1'
        self.model_name = 'nsi319/legal-pegasus'
        self.max_length = 1024
        self.min_summary_length = 30
        self.max_summary_length = 300
        self.max_concurrent_tasks = 10
