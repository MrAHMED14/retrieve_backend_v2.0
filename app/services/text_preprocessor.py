import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Download 'stopwords' only if missing
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    processed = []
    for word in words:
        if word not in stop_words and len(word) > 2:
            processed.append(stemmer.stem(word))
    return " ".join(processed)