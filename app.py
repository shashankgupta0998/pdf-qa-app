import os
import sys
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
)


# ─── Agent 1: Ingestion ───────────────────────────────────────────────────
def ingestion_agent(pdf_path: str) -> Chroma:
    print(f"\n📥 Ingestion Agent: Loading '{pdf_path}'...")
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"   Loaded {len(docs)} pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)
    print(f"   Split into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_DIR
    )
    print(f"   Stored in ChromaDB.")
    return vectorstore


# ─── Agent 2: Retrieval ───────────────────────────────────────────────────
def retrieval_agent(vectorstore: Chroma, question: str) -> list:
    print(f"\n🔎 Retrieval Agent: Searching top 3 chunks...")
    results = vectorstore.similarity_search(question, k=3)
    print(f"   Retrieved {len(results)} chunks.")
    return results


# ─── Agent 3: Answer ──────────────────────────────────────────────────────
def answer_agent(question: str, chunks: list) -> str:
    print(f"\n💡 Answer Agent: Generating initial answer...")
    context = "\n\n".join(
        f"[Chunk {i+1}]\n{c.page_content}" for i, c in enumerate(chunks)
    )
    messages = [
        SystemMessage(content=(
            "You answer questions strictly from the provided PDF context. "
            "If the answer is not in the context, say so. Be concise and factual."
        )),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    answer = model.invoke(messages).content
    print(f"   Initial answer drafted.")
    return answer


# ─── Agent 4: Critic ──────────────────────────────────────────────────────
def critic_agent(question: str, chunks: list, answer: str) -> str:
    print(f"\n🧐 Critic Agent: Reviewing answer for gaps/inaccuracies...")
    context = "\n\n".join(
        f"[Chunk {i+1}]\n{c.page_content}" for i, c in enumerate(chunks)
    )
    messages = [
        SystemMessage(content=(
            "You are a strict fact-checker. Compare the answer to the source chunks. "
            "List specific gaps, unsupported claims, or inaccuracies. "
            "If the answer is fully correct and complete, say 'NO ISSUES'. "
            "Otherwise, provide a short bulleted critique."
        )),
        HumanMessage(content=(
            f"Question: {question}\n\nSource Chunks:\n{context}\n\n"
            f"Answer to review:\n{answer}"
        )),
    ]
    critique = model.invoke(messages).content
    print(f"   Critique complete.")
    return critique


# ─── Agent 5: Refiner ─────────────────────────────────────────────────────
def refiner_agent(question: str, chunks: list, answer: str, critique: str) -> str:
    print(f"\n✨ Refiner Agent: Producing final answer...")
    context = "\n\n".join(
        f"[Chunk {i+1}]\n{c.page_content}" for i, c in enumerate(chunks)
    )
    messages = [
        SystemMessage(content=(
            "You produce a final, high-accuracy answer grounded in the source chunks. "
            "Incorporate the critic's feedback to fix gaps or inaccuracies. "
            "Stay strictly faithful to the context."
        )),
        HumanMessage(content=(
            f"Question: {question}\n\nSource Chunks:\n{context}\n\n"
            f"Initial Answer:\n{answer}\n\nCritic Feedback:\n{critique}\n\n"
            "Write the final refined answer:"
        )),
    ]
    refined = model.invoke(messages).content
    print(f"   Final answer ready.")
    return refined


# ─── Agent 6: Orchestrator ────────────────────────────────────────────────
def orchestrator(vectorstore: Chroma, question: str) -> str:
    print(f"\n{'='*60}")
    print(f"🧠 Orchestrator: Pipeline for: {question!r}")
    print(f"{'='*60}")
    chunks = retrieval_agent(vectorstore, question)
    initial = answer_agent(question, chunks)
    critique = critic_agent(question, chunks, initial)
    final = refiner_agent(question, chunks, initial, critique)
    print(f"\n{'='*60}")
    print("Orchestrator: Pipeline complete.")
    print(f"{'='*60}")
    return final


# ─── CLI ──────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python app.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    vectorstore = ingestion_agent(pdf_path)

    print("\nReady. Type a question (or 'quit' to exit).")
    while True:
        try:
            q = input("\n❓ Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"quit", "exit", "q"}:
            break
        final = orchestrator(vectorstore, q)
        print(f"\n📄 FINAL ANSWER:\n{final}\n")

    print("Goodbye.")


if __name__ == "__main__":
    main()
