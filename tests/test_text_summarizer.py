
import unittest
import asyncio
from src.text_summarizer import summarize_text

class TestTextSummarizer(unittest.TestCase):
    def test_summarize_text(self):
        sample_text = "This is a long piece of text that needs to be summarized. " * 20
        summary = asyncio.run(summarize_text(sample_text))
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) < len(sample_text))

if __name__ == '__main__':
    unittest.main()