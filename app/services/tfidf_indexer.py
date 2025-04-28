from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.services.text_preprocessor import preprocess_text


class TFIDFIndexer:
    def __init__(self):
        self.documents = []
        self.original_texts = []    
        self.filenames = []
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None

    def add_document(self, text, filename):
        # Check if file is already indexed
        if filename in self.filenames:
            print(f"File '{filename}' is already indexed.")
            return  # Do not add the file again

        processed_text = preprocess_text(text)
        self.documents.append(processed_text)
        self.original_texts.append(text)  # Save original
        self.filenames.append(filename)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        print(f"File '{filename}' added to the index.")

    def search(self, query):

        if len(self.documents) == 0:
            return []
    
        processed_query = preprocess_text(query)
        query_vec = self.vectorizer.transform([processed_query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        ranked_indices = scores.argsort()[::-1]
        
        results = []
        for idx in ranked_indices:
            if scores[idx] > 0:
                snippet = self.extract_snippet(self.original_texts[idx], query)
                results.append({
                    "filename": self.filenames[idx],
                    "score": float(scores[idx]),
                    "snippet": snippet
                })
        return results

    def extract_snippet(self, text, query, window_size=30):
        """Return a secure highlighted snippet around matched query words."""
        import html

        query_words = query.lower().split()
        words = text.split()

        escaped_words = [html.escape(w) for w in words]
        
        
        for i, word in enumerate(words):
            if any(qw in word.lower() for qw in query_words):
                start = max(i - window_size//2, 0)
                end = min(i + window_size//2, len(words))
                snippet_words = escaped_words[start:end]

                highlighted_snippet = []
                for w in snippet_words:
                    if any(qw in w.lower() for qw in query_words):
                        highlighted_snippet.append(f"<mark>{w}</mark>")
                    else:
                        highlighted_snippet.append(w)
                
                snippet = " ".join(highlighted_snippet)
                return f"... {snippet} ..."
        
        # fallback
        safe_text = html.escape(text)
        return safe_text[:window_size*2] + "..."

    def remove_file(self, filename):
        """Remove a specific file from the index."""
        if filename not in self.filenames:
            raise ValueError(f"File '{filename}' not found in the index.")
        
        file_index = self.filenames.index(filename)
        
        self.filenames.pop(file_index)
        self.documents.pop(file_index)
        self.original_texts.pop(file_index)
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        
    def remove_all_files(self):
        """Clear the entire index."""
        self.filenames.clear()
        self.documents.clear()
        self.original_texts.clear()
        self.tfidf_matrix = None
