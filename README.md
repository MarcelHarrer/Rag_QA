# RAG QA System

This project implements a Retrieval-Augmented Generation (RAG) system for question answering. It uses a combination of document retrieval and a question-answering model to provide accurate answers based on the content of PDF files.

## Features

- Extracts text from PDF files.
- Builds a FAISS index for efficient document retrieval.
- Uses a powerful embedding model (`all-mpnet-base-v2`) for semantic search.
- Answers questions using a robust QA model (`deepset/roberta-base-squad2`).

## Requirements

Install the required Python libraries:

```bash
pip install PyPDF2 sentence-transformers faiss-cpu transformers
```

## Usage

1. **Run the script**:
   ```bash
   python main.py
   ```

2. **Options**:
   - **Load PDF**: Provide the path to a PDF file to extract text.
   - **Build Index**: Create a FAISS index for the loaded text.
   - **Ask a Question**: Enter a question to retrieve relevant information and get an answer.
   - **Exit**: Exit the program.

## Example

1. Load a PDF:
   ```
   Enter the path to the PDF file: example.pdf
   ```

2. Build the index:
   ```
   Index built successfully.
   ```

3. Ask a question:
   ```
   Enter your question: What is the main topic of the document?
   Answer: The main topic is...
   ```

## Notes

- Ensure the PDF file contains readable text (not scanned images).
- The system works best with well-structured documents.

## License

This project is licensed under the MIT License.