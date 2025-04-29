from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.services.text_preprocessor import preprocess_text
import uuid

class TFIDFIndexer:
    def __init__(self):
        self.documents_info = []  # Each item: {id, filename, text}
        self.processed_documents = []
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None

    def add_document(self, text, filename):
        doc_id = str(uuid.uuid4())

        processed_text = preprocess_text(text)
        self.documents_info.append({"id": doc_id, "filename": filename, "text": text})
        self.processed_documents.append(processed_text)

        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_documents)
        print(f"File '{filename}' added with ID '{doc_id}'.")

        return doc_id

    def search(self, query):
        if not self.documents_info:
            return []

        processed_query = preprocess_text(query)
        query_vec = self.vectorizer.transform([processed_query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        ranked_indices = scores.argsort()[::-1]

        results = []
        for idx in ranked_indices:
            if scores[idx] > 0:
                snippet = self.extract_snippet(self.documents_info[idx]["text"], query)
                results.append({
                    "id": self.documents_info[idx]["id"],
                    "filename": self.documents_info[idx]["filename"],
                    "score": float(scores[idx]),
                    "snippet": snippet
                })
        return results

    def extract_snippet(self, text, query, window_size=30):
        import html

        query_words = query.lower().split()
        words = text.split()
        escaped_words = [html.escape(w) for w in words]

        for i, word in enumerate(words):
            if any(qw in word.lower() for qw in query_words):
                start = max(i - window_size // 2, 0)
                end = min(i + window_size // 2, len(words))
                snippet_words = escaped_words[start:end]

                highlighted_snippet = []
                for w in snippet_words:
                    if any(qw in w.lower() for qw in query_words):
                        highlighted_snippet.append(f"<mark>{w}</mark>")
                    else:
                        highlighted_snippet.append(w)

                snippet = " ".join(highlighted_snippet)
                return f"... {snippet} ..."

        safe_text = html.escape(text)
        return safe_text[:window_size*2] + "..."

    def remove_file(self, doc_id):
        for idx, doc in enumerate(self.documents_info):
            if doc["id"] == doc_id:
                self.documents_info.pop(idx)
                self.processed_documents.pop(idx)
                if self.processed_documents:
                    self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_documents)
                else:
                    self.tfidf_matrix = None
                return
        raise ValueError(f"File with ID '{doc_id}' not found.")

    def remove_all_files(self):
        self.documents_info.clear()
        self.processed_documents.clear()
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
