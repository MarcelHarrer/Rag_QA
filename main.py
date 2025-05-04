import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# Initialize global variables
model = SentenceTransformer('all-mpnet-base-v2')  # More robust embedding model
index = None
documents = []
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")  # Improved QA model

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
            print(f"Loaded {len(documents)} lines from {file_path}.")
    except Exception as e:
        print(f"Error loading PDF: {e}")

def build_index():
    """Build a FAISS index from the loaded documents."""
    global index
    if not documents:
        print("No documents loaded. Please load a PDF first.")
        return
    embeddings = model.encode(documents)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    print("Index built successfully.")

def retrieve(query, top_k=5):
    """Retrieve top-k relevant chunks for a query."""
    if index is None:
        print("Index not built. Please build the index first.")
        return []
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]

def answer_question(query):
    """Answer a question using the RAG approach."""
    relevant_chunks = retrieve(query)
    if not relevant_chunks:
        return "No relevant information found."
    context = "\n".join(relevant_chunks)
    response = qa_pipeline(question=query, context=context)
    return response['answer']

def main():
    while True:
        print("\nOptions:")
        print("1. Load PDF")
        print("2. Build Index")
        print("3. Ask a Question")
        print("4. Exit")
        choice = input("Enter your choice: ")
        if choice == "1":
            file_path = input("Enter the path to the PDF file: ")
            load_pdf(file_path)
        elif choice == "2":
            build_index()
        elif choice == "3":
            query = input("Enter your question: ")
            answer = answer_question(query)
            print(f"Answer: {answer}")
        elif choice == "4":
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
