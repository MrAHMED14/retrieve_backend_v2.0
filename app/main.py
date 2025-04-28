from fastapi import FastAPI, UploadFile, File, HTTPException # type: ignore
from fastapi.responses import JSONResponse # type: ignore
from app.services.file_handler import extract_text
from app.services.text_preprocessor import preprocess_text
from app.services.tfidf_indexer import TFIDFIndexer
from app.storage.database import save_index, load_index
import os

app = FastAPI()

# Initialize the indexer
indexer = TFIDFIndexer()

# Load existing index if exists
if os.path.exists("tfidf_index.pkl"):
    indexer = load_index("tfidf_index.pkl")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    text = extract_text(content, file.filename)
    clean_text = preprocess_text(text)
    indexer.add_document(clean_text, file.filename)
    save_index(indexer, "tfidf_index.pkl")
    return {"message": "File indexed successfully", "filename": file.filename}

@app.post("/search")
async def search_documents(query: str):
    results = indexer.search(query)
    return {"results": results}

@app.get("/files")
async def list_files():
    filenames = indexer.filenames
    return JSONResponse(content={"files": filenames})

@app.delete("/remove-file/{filename}")
def remove_file(filename: str):
    try:
        indexer.remove_file(filename)
        return JSONResponse(content={"message": f"File '{filename}' removed successfully!"})
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.delete("/remove-all-files")
def remove_all_files():
    indexer.remove_all_files()
    return JSONResponse(content={"message": "All files removed successfully!"})