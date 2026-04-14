# PDF QA App — Multi-Agent RAG

A command-line **multi-agent** Retrieval-Augmented Generation (RAG) application that answers natural-language questions about any PDF. Instead of a single LLM call, it runs a pipeline of six specialized agents that ingest, retrieve, answer, critique, and refine — producing higher-accuracy, source-grounded answers.

## Architecture

```
                         ┌─────────────────────────┐
   PDF ───────────────►  │  1. Ingestion Agent     │
                         │  PyPDF → Splitter →     │
                         │  HF Embeddings → Chroma │
                         └───────────┬─────────────┘
                                     │ vectorstore
                                     ▼
                         ┌─────────────────────────┐
   Question ──────────►  │  6. Orchestrator        │
                         │  (runs agents 2→5)      │
                         └───────────┬─────────────┘
                                     │
                                     ▼
                         ┌─────────────────────────┐
                         │  2. Retrieval Agent     │
                         │  similarity_search(k=3) │
                         └───────────┬─────────────┘
                                     │ top-3 chunks
                                     ▼
                         ┌─────────────────────────┐
                         │  3. Answer Agent        │
                         │  Groq llama-3.1-8b      │
                         │  → initial answer       │
                         └───────────┬─────────────┘
                                     │
                                     ▼
                         ┌─────────────────────────┐
                         │  4. Critic Agent        │
                         │  fact-checks answer     │
                         │  vs. source chunks      │
                         └───────────┬─────────────┘
                                     │ critique
                                     ▼
                         ┌─────────────────────────┐
                         │  5. Refiner Agent       │
                         │  merges answer+critique │
                         │  → final answer         │
                         └───────────┬─────────────┘
                                     │
                                     ▼
                                Final Answer
```

## The Six Agents

1. **Ingestion Agent** — Loads the PDF with PyPDF, splits it into 1000-char chunks (150 overlap), embeds with HuggingFace `all-MiniLM-L6-v2`, and persists to ChromaDB.
2. **Retrieval Agent** — Performs similarity search against ChromaDB and returns the top 3 most relevant chunks for the question.
3. **Answer Agent** — Sends the question + retrieved chunks to Groq `llama-3.1-8b-instant` to draft an initial grounded answer.
4. **Critic Agent** — Independently reviews the draft answer against the source chunks, flagging gaps, unsupported claims, or inaccuracies (or returns `NO ISSUES`).
5. **Refiner Agent** — Takes the initial answer plus the critic's feedback and produces a final, high-accuracy answer faithful to the context.
6. **Orchestrator** — Runs agents 2 → 5 in sequence for each question, logs stage progress, and returns the refined answer.

## Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/)

## Installation

```bash
git clone https://github.com/shashankgupta0998/pdf-qa-app.git
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

The app ingests the PDF once, then starts an interactive question loop. Each question runs through the full orchestrated agent pipeline, with per-agent progress printed to the console. Type `quit` or `exit` to stop.

## Tech Stack

- [LangChain](https://www.langchain.com/) — Agent orchestration framework
- [Groq](https://groq.com/) — LLM inference (`llama-3.1-8b-instant`)
- [ChromaDB](https://www.trychroma.com/) — Vector database
- [HuggingFace](https://huggingface.co/) — Sentence-transformer embeddings
- [PyPDF](https://pypdf.readthedocs.io/) — PDF parsing
