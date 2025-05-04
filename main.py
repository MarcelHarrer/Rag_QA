import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Initialize global variables
model = SentenceTransformer('all-mpnet-base-v2')  
index = None
documents = []
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

app = FastAPI()
templates = Jinja2Templates(directory="c:/Rag_QA/templates")
app.mount("/static", StaticFiles(directory="c:/Rag_QA/static"), name="static")

def load_pdf_from_file(file: UploadFile):
    """Extract text from an uploaded PDF file."""
    global documents
    try:
        reader = PyPDF2.PdfReader(file.file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        documents = text.split("\n")  # Split into chunks by lines
        return {"message": f"Loaded {len(documents)} lines from the uploaded PDF."}
    except Exception as e:
        return {"error": f"Error loading PDF: {e}"}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-pdf/")
async def upload_pdf(request: Request, files: list[UploadFile]):
    """Handle multiple PDF uploads."""
    global documents
    all_messages = []
    for file in files:
        result = load_pdf_from_file(file)
        if "message" in result:
            all_messages.append(result["message"])
        if "error" in result:
            all_messages.append(result["error"])
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "message": " | ".join(all_messages)}
    )

@app.post("/build-index/")
async def build_index(request: Request):
    """Handle index building."""
    global index
    if not documents:
        return templates.TemplateResponse("index.html", {"request": request, "error": "No documents loaded. Please upload a PDF first."})
    embeddings = model.encode(documents)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return templates.TemplateResponse("index.html", {"request": request, "message": "Index built successfully."})

def retrieve(query: str, top_k: int = 5):
    """Retrieve top-k relevant chunks for a query."""
    if index is None:
        return {"error": "Index not built. Please build the index first."}
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]

@app.post("/ask-question/")
async def ask_question(request: Request, query: str = Form(...)):
    """Handle question answering."""
    relevant_chunks = retrieve(query)
    if isinstance(relevant_chunks, dict):  # Check for errors
        return templates.TemplateResponse("index.html", {"request": request, "error": relevant_chunks["error"]})
    context = "\n".join(relevant_chunks)
    response = qa_pipeline(question=query, context=context)
    return templates.TemplateResponse("index.html", {"request": request, "answer": response['answer']})
