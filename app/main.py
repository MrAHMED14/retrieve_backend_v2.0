from fastapi import FastAPI, UploadFile, File, HTTPException # type: ignore
from fastapi.responses import JSONResponse # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from app.services.file_handler import extract_text
from app.services.text_preprocessor import preprocess_text
from app.services.tfidf_indexer import TFIDFIndexer
from app.storage.database import save_index, load_index
from typing import List
import os

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the indexer
indexer = TFIDFIndexer()

# Load existing index if exists
if os.path.exists("tfidf_index.pkl"):
    indexer = load_index("tfidf_index.pkl")

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    uploaded = []

    for file in files:
        content = await file.read()
        text = extract_text(content, file.filename)
        doc_id = indexer.add_document(text, file.filename)
        if doc_id: 
            uploaded.append({"filename": file.filename, "id": doc_id})

    save_index(indexer, "tfidf_index.pkl")

    return {"message": f"{len(uploaded)} new files indexed successfully!", "files": uploaded}

@app.post("/search")
async def search_documents(query: str):
    results = indexer.search(query)
    return {"results": results}

@app.get("/files")
async def list_files():
    files = [{"id": doc["id"], "filename": doc["filename"]} for doc in indexer.documents_info]
    return JSONResponse(content={"files": files})

@app.delete("/remove-file/{doc_id}")
def remove_file(doc_id: str):
    try:
        indexer.remove_file(doc_id)
        save_index(indexer, "tfidf_index.pkl")
        return JSONResponse(content={"message": f"File with ID '{doc_id}' removed successfully!"})
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.delete("/remove-all-files")
def remove_all_files():
    indexer.remove_all_files()
    save_index(indexer, "tfidf_index.pkl")
    return JSONResponse(content={"message": "All files removed successfully!"}) 