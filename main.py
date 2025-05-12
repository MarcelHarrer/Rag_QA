from flask import Flask, request, jsonify, render_template
import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from werkzeug.utils import secure_filename

# Initialize global variables
model = SentenceTransformer('all-mpnet-base-v2')  
index = None
documents = []
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_pdf(file_path):
    """Extract text from a PDF file."""
    global documents
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            documents = text.split("\n")  # Split into chunks by lines
            return f"Loaded {len(documents)} lines from {file_path}."
    except Exception as e:
        return f"Error loading PDF: {e}"

def build_index():
    """Build a FAISS index from the loaded documents."""
    global index
    if not documents:
        return "No documents loaded. Please load a PDF first."
    embeddings = model.encode(documents)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return "Index built successfully."

def retrieve(query, top_k=5):
    """Retrieve top-k relevant chunks for a query."""
    if index is None:
        return []
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]

def answer_question(query):
    """Answer a question using the RAG approach and include relevant snippets."""
    relevant_chunks = retrieve(query)
    if not relevant_chunks:
        return {"answer": "No relevant information found.", "snippets": []}
    context = "\n".join(relevant_chunks)
    response = qa_pipeline(question=query, context=context)
    return {"answer": response['answer'], "snippets": relevant_chunks}

@app.before_request
def clear_uploads_on_reload():
    """Clear the uploads folder when the home page is accessed."""
    if request.endpoint == 'home':
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/load_pdf', methods=['POST'])
def api_load_pdf():
    if 'files' not in request.files:
        return jsonify({"message": "No files part in the request."})
    files = request.files.getlist('files')
    if not files:
        return jsonify({"message": "No files selected."})
    
    all_messages = []
    for file in files:
        if file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            message = load_pdf(file_path)
            all_messages.append(message)
    
    return jsonify({"message": " | ".join(all_messages)})

@app.route('/build_index', methods=['POST'])
def api_build_index():
    message = build_index()
    return jsonify({"message": message})

@app.route('/ask_question', methods=['POST'])
def api_ask_question():
    query = request.form.get('query')
    result = answer_question(query)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
