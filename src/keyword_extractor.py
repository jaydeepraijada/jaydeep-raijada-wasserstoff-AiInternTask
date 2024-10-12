import yake
from nltk.corpus import stopwords
import logging

logger = logging.getLogger(__name__)

def extract_keywords(text):
    try:
        stop_words = set(stopwords.words('english'))
        
        custom_kw_extractor = yake.KeywordExtractor(
            lan="en", 
            n=1,
            dedupLim=0.9,
            dedupFunc='seqm',
            windowsSize=1,
            top=10,
            features=None,
            stopwords=stop_words
        )
        
        keywords = custom_kw_extractor.extract_keywords(text)
        
        return [kw for kw, _ in keywords]
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        return []