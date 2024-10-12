import unittest
from src.keyword_extractor import extract_keywords

class TestKeywordExtractor(unittest.TestCase):
    def test_extract_keywords(self):
        sample_text = "This is a sample text about artificial intelligence and machine learning."
        keywords = extract_keywords(sample_text)
        self.assertIsInstance(keywords, list)
        self.assertTrue(len(keywords) > 0)
        self.assertTrue("artificial intelligence" in keywords or "machine learning" in keywords)

if __name__ == '__main__':
    unittest.main()