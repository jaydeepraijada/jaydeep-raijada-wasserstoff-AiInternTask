import asyncio
import logging
from src.pdf_processor import process_pdfs_in_folder
from src.database_handler import init_mongodb
from src.performance_monitor import PerformanceMonitor
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration
PDF_FOLDER_PATH = r"C:\Users\acer\Desktop\Task_pdfs"
MONGODB_URI = 'mongodb://localhost:27017/'
DATABASE_NAME = 'pdf_database'
COLLECTION_NAME = 'pdf_documents'
MODEL_NAME = 'nsi319/legal-pegasus'
MAX_LENGTH = 1024
MIN_SUMMARY_LENGTH = 30
MAX_SUMMARY_LENGTH = 300

# Set up logging
logging.basicConfig(filename='logs/processing.log', level=logging.INFO)
logger = logging.getLogger(__name__)

async def init_mongodb():
    uri = os.getenv('MONGODB_URI')
    # Create a new client and connect to the server
    client = AsyncIOMotorClient(uri, server_api=ServerApi('1'))
    # Send a ping to confirm a successful connection
    try:
        await client.admin.command('ping')
        print("Successfully connected to MongoDB")
        return client
    except Exception as e:
        print(e)
        return None

async def main():
    client = await init_mongodb()
    if client is None:
        print("Failed to connect to MongoDB")
        return

    db = client.get_database(DATABASE_NAME)
    collection = db.get_collection(COLLECTION_NAME)

    config = {
        'pdf_folder_path': PDF_FOLDER_PATH,
        'mongodb_uri': MONGODB_URI,
        'database_name': DATABASE_NAME,
        'collection_name': COLLECTION_NAME,
        'model_name': MODEL_NAME,
        'max_length': MAX_LENGTH,
        'min_summary_length': MIN_SUMMARY_LENGTH,
        'max_summary_length': MAX_SUMMARY_LENGTH
    }
    
    performance_monitor = PerformanceMonitor()
    
    # Create a semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(10)  # or higher, depending on your system's capabilities
    
    results = await process_pdfs_in_folder(config['pdf_folder_path'], config, semaphore, performance_monitor)
    
    report = performance_monitor.generate_report(results)
    performance_monitor.save_report(report)

    #calculate total time and successful processes
    total_time = sum(result[3] for result in results)
    successful_processes = [r for r in results if r[1]]

    print(f"\nProcessed {len(successful_processes)} out of {len(results)} PDFs successfully")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average processing time per document: {total_time/len(results):.2f} seconds")

    #print results
    for pdf_file, summary, keywords, proc_time in results:
        if summary:
            print(f"\nProcessed {pdf_file} in {proc_time:.2f} seconds:")
            print(f"Summary: {summary}")
            print(f"Keywords: {', '.join(keywords)}\n")
        else:
            print(f"\nFailed to process {pdf_file}\n")

    # Don't forget to close the client when you're done
    client.close()

if __name__ == "__main__":
    asyncio.run(main())
