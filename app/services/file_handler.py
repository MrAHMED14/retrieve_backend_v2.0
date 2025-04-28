import fitz  # PyMuPDF

def extract_text(file_bytes, filename):
    if filename.endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    elif filename.endswith(".txt"):
        return file_bytes.decode('utf-8')
    else:
        raise ValueError("Unsupported file type")
