import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rag_pipeline import RAGPipeline

class QueryRequest(BaseModel):
    question: str

app = FastAPI()
pipeline = RAGPipeline()


@app.get("/")
def root():
    return {"message" : "Fast API is running"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    file_path = os.path.join("./source", file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    pipeline.ingest(file_path)

    return JSONResponse({
        "message" : "PDF uploaded successfully"
    }, status_code=200)

@app.post("/query")
async def query_rag(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        answer = await pipeline.query(question)
        return {"question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))