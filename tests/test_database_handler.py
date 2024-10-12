import unittest
import asyncio
from src.database_handler import update_mongodb

class TestDatabaseHandler(unittest.TestCase):
    def test_update_mongodb(self):
        sample_data = {
            'file_path': '/path/to/sample.pdf',
            'summary': 'This is a sample summary.',
            'keywords': ['sample', 'test', 'keywords']
        }
        result = asyncio.run(update_mongodb(sample_data['file_path'], sample_data['summary'], sample_data['keywords']))
        self.assertIsNotNone(result)
        # Add more assertions based on the expected behavior of your update_mongodb function

if __name__ == '__main__':
    unittest.main()