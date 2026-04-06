# PDF QA App

A command-line Retrieval-Augmented Generation (RAG) application that lets you ask natural language questions about any PDF document. It extracts text from the PDF, builds a vector index, and uses an LLM to answer your questions based on the document content.

## How It Works

1. Loads and parses the PDF using PyPDF
2. Splits the text into chunks (1000 chars, 200 overlap)
3. Generates embeddings with HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
4. Stores vectors in a local ChromaDB database
5. For each question, retrieves relevant chunks and sends them to Groq's `llama-3.3-70b-versatile` model to generate an answer

## Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/)

## Installation

```bash
git clone https://github.com/shashankgupta/pdf-qa-app.git
cd pdf-qa-app

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```
GROQ_API_KEY="your-groq-api-key"
```

## Usage

```bash
python app.py <path-to-pdf>
```

Example:

```bash
python app.py report.pdf
```

The app will process the PDF and start an interactive session. Type your questions and press Enter. Type `quit` or `exit` to stop.

## Tech Stack

- [LangChain](https://www.langchain.com/) - Orchestration framework
- [Groq](https://groq.com/) - LLM inference (Llama 3.3 70B)
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [HuggingFace](https://huggingface.co/) - Text embeddings
- [PyPDF](https://pypdf.readthedocs.io/) - PDF parsing
