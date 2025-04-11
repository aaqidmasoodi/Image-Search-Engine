import re
import os
import nltk
import ssl
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from typing import List

# SSL certificate fix
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
        self.stop_words = set(stopwords.words('english')) - {
            'image', 'photo', 'diagram', 'graph', 'map', 'chart'
        }

    def clean_wikipedia_text(self, text: str) -> str:
        """Special cleaning for Wikipedia artifacts"""
        if not text: 
            return ""
            
        # Remove templates, citations, section headers
        text = re.sub(r'\[\d+\]|\{\{.*?\}\}|==.*?==|\'\'\'', '', text)
        
        # Remove non-informative phrases
        stop_phrases = ['portal', 'category', 'glossary', 'img', 'file', 'icon']
        for phrase in stop_phrases:
            text = re.sub(rf'\b{phrase}\b', '', text, flags=re.IGNORECASE)
            
        return text.strip()

    def preprocess_text(self, text: str, stem: bool = True) -> List[str]:
        """Full preprocessing pipeline"""
        text = self.clean_wikipedia_text(text)
        if not text:
            return []
            
        # Tokenization and filtering
        words = word_tokenize(text.lower())
        processed_words = [
            self.stemmer.stem(w) if (stem and len(w) > 4) else w
            for w in words
            if w.isalnum() and w not in self.stop_words
        ]
        return processed_words

# Global instance
preprocessor = TextPreprocessor()
preprocess_text = preprocessor.preprocess_text