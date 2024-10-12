import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi

from src.config import Config
from src.pdf_processor import process_pdfs_in_folder
from src.performance_monitor import PerformanceMonitor
from src.database_handler import DatabaseHandler

# Set up logging
logging.basicConfig(filename='logs/processing.log', level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessingApp:
    def __init__(self, config: Config):
        self.config = config
        self.db_handler = None
        self.performance_monitor = PerformanceMonitor()

    async def init_mongodb(self):
        """Initialize MongoDB connection and create DatabaseHandler."""
        try:
            client = AsyncIOMotorClient(self.config.mongodb_uri, server_api=ServerApi('1'))
            await client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
            self.db_handler = DatabaseHandler(client, self.config.database_name, self.config.collection_name)
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    async def run(self):
        """Main execution method for processing PDFs."""
        await self.init_mongodb()
        
        # Create a semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        
        print(f"Starting to process PDFs in folder: {self.config.pdf_folder_path}")
        results = await process_pdfs_in_folder(
            self.config.pdf_folder_path,
            self.config,
            self.db_handler,
            semaphore,
            self.performance_monitor
        )
        
        self.print_results(results)
        self.save_performance_report(results)

    def print_results(self, results):
        """Print a summary of processing results."""
        total_time = sum(result['processing_time'] for result in results)
        successful_processes = [r for r in results if r['summary']]

        print(f"\nProcessed {len(successful_processes)} out of {len(results)} PDFs successfully")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average processing time per document: {total_time/len(results):.2f} seconds")

        for result in results:
            print(f"\nProcessed: {result['pdf_file']}")
            print(f"Processing time: {result['processing_time']:.2f} seconds")
            if result['summary']:
                print(f"Summary: {result['summary'][:100]}...")  # Print first 100 chars of summary
                print(f"Keywords: {', '.join(result['keywords'][:5])}")  # Print first 5 keywords
            else:
                print(f"Failed to process. Error: {result.get('error', 'Unknown error')}")

    def save_performance_report(self, results):
        """Generate and save performance report."""
        report = self.performance_monitor.generate_report(results)
        self.performance_monitor.save_report(report)
        print(f"Performance report saved to {self.performance_monitor.report_filename}")

async def main():
    """Main entry point of the application."""
    config = Config()
    app = PDFProcessingApp(config)
    
    try:
        print("Starting PDF processing application...")
        await app.run()
        print("PDF processing completed successfully.")
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        logger.error(f"An error occurred during execution: {e}")
    finally:
        if app.db_handler:
            await app.db_handler.close()

if __name__ == "__main__":
    asyncio.run(main())
