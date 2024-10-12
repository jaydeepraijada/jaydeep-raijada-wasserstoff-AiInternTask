import unittest
import asyncio
import os
from src.pdf_processor import extract_text_from_pdf, process_pdfs_in_folder
from src.performance_monitor import PerformanceMonitor

class TestPDFProcessor(unittest.TestCase):
    def setUp(self):
        self.sample_folder = 'path/to/sample/folder'
        self.config = {
            'pdf_folder_path': self.sample_folder,
            'mongodb_uri': 'mongodb://localhost:27017/',
            'database_name': 'test_db',
            'collection_name': 'test_collection',
            'model_name': 'nsi319/legal-pegasus',
            'max_length': 1024,
            'min_summary_length': 30,
            'max_summary_length': 300
        }

    def test_extract_text_from_pdf(self):
        # You'll need a sample PDF file for this test
        sample_pdf_path = os.path.join(self.sample_folder, 'sample.pdf')
        text = asyncio.run(extract_text_from_pdf(sample_pdf_path))
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)

    def test_process_pdfs_in_folder(self):
        results = asyncio.run(process_pdfs_in_folder(self.sample_folder, self.config))
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)
        
        # Check if performance report was generated
        self.assertTrue(os.path.exists('performance_report.txt'))
        
        # Check the structure of each result
        for result in results:
            self.assertEqual(len(result), 4)  # (pdf_file, summary, keywords, processing_time)
            self.assertIsInstance(result[0], str)  # pdf_file
            self.assertIsInstance(result[1], str)  # summary
            self.assertIsInstance(result[2], list)  # keywords
            self.assertIsInstance(result[3], float)  # processing_time

if __name__ == '__main__':
    unittest.main()
