from sklearn.feature_extraction.text import TfidfVectorizer

class TextFeatureExtractor:
    def __init__(self):
        # Initialize vectorizers for different text components
        self.description_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        
        # Skill extraction and matching tools
        self.skill_vectorizer = TfidfVectorizer(
            binary=True,  
            lowercase=True,
            token_pattern=r'(?u)\b[A-Za-z0-9+#]+\b'
        )
