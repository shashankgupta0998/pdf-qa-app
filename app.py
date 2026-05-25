import os
import glob
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class AgentResult:
    status: Literal["success", "error", "no_results"]
    data: Any = None
    error: str | None = None
    is_retryable: bool = False
    metadata: dict = field(default_factory=dict)


load_dotenv()

CHROMA_DIR = "chroma_db"
DOCS_DIR = "documents"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.35
MAX_REFINE_RETRIES = 2


def build_context(chunks: list) -> str:
    sorted_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)
    sections = []
    for i, (doc, score) in enumerate(sorted_chunks):
        label = "[Most Relevant]" if i == 0 else "[Supporting]"
        src = os.path.basename(doc.metadata.get("source", "unknown"))
        pg = doc.metadata.get("page", "?")
        sections.append(f"{label} (Source: {src}, p.{pg}, score: {score:.2f})\n{doc.page_content}")
    return "\n\n".join(sections)


model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
)


# ─── Agent 1: Ingestion ───────────────────────────────────────────────────
def ingestion_agent() -> Chroma:
    os.makedirs(DOCS_DIR, exist_ok=True)
    pdf_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))

    if not pdf_files:
        print(f"\n❌ No PDF files found in '{DOCS_DIR}/' folder.")
        print(f"   Place your PDF files in the '{DOCS_DIR}/' folder and restart.")
        raise SystemExit(1)

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    print(f"\n📥 Ingestion Agent: Loading {len(pdf_files)} PDF(s) from '{DOCS_DIR}/'...")
    all_docs = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print(f"   ✓ {os.path.basename(pdf_path)} — {len(docs)} pages")
        all_docs.extend(docs)

    print(f"   Total: {len(all_docs)} pages across {len(pdf_files)} file(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    )
    chunks = splitter.split_documents(all_docs)
    print(f"   Split into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_DIR
    )
    print(f"   Stored in ChromaDB.")
    return vectorstore


# ─── Agent 2: Retrieval ───────────────────────────────────────────────────
def retrieval_agent(vectorstore: Chroma, question: str) -> AgentResult:
    print(f"\n🔎 Retrieval Agent: Searching top 3 chunks...")
    try:
        scored_results = vectorstore.similarity_search_with_relevance_scores(question, k=3)
    except Exception as e:
        print(f"   ❌ Retrieval failed: {e}")
        return AgentResult(
            status="error",
            error=str(e),
            is_retryable=True,
            metadata={"agent": "retrieval"},
        )

    relevant = [(doc, score) for doc, score in scored_results if score >= SIMILARITY_THRESHOLD]

    if not relevant:
        scores_str = ", ".join(f"{s:.2f}" for _, s in scored_results)
        print(f"   ⚠️  All chunks below threshold ({SIMILARITY_THRESHOLD}). Scores: [{scores_str}]")
        return AgentResult(
            status="no_results",
            data=[],
            metadata={"agent": "retrieval", "reason": "below_threshold", "scores": [s for _, s in scored_results]},
        )

    print(f"   Retrieved {len(relevant)} chunks above threshold ({SIMILARITY_THRESHOLD}).")
    for doc, score in relevant:
        src = doc.metadata.get("source", "unknown")
        pg = doc.metadata.get("page", "?")
        print(f"     • {os.path.basename(src)} p.{pg} — score {score:.2f}")

    return AgentResult(
        status="success",
        data=relevant,
        metadata={"agent": "retrieval", "count": len(relevant)},
    )


# ─── Agent 3: Answer ──────────────────────────────────────────────────────
def answer_agent(question: str, chunks: list, history: list) -> AgentResult:
    print(f"\n💡 Answer Agent: Generating initial answer...")
    context = build_context(chunks)
    history_text = ""
    if history:
        history_text = "Conversation History:\n" + "\n".join(
            f"Q: {h['question']}\nA: {h['answer']}" for h in history
        ) + "\n\n"
    messages = [
        SystemMessage(content=(
            "You answer questions strictly from the provided PDF context. "
            "If the answer is not in the context, say so. Be concise and factual. "
            "Use the conversation history to understand follow-up questions."
            "\n\nIf a specific detail (date, number, name) is not stated in the context, "
            "say 'unclear from the source' or 'not stated in the provided documents'. "
            "Do NOT guess, fabricate, or supplement with general knowledge. "
            "It is better to say 'unclear' than to risk an inaccurate answer."
            "\n\nExamples of how to handle borderline cases:\n\n"
            "Example 1 — Partially answerable:\n"
            "Context: 'Revenue grew 15% in Q3 2024.'\n"
            "Question: 'What was the revenue growth for each quarter?'\n"
            "Answer: 'The context only mentions Q3 2024, which saw 15% revenue growth. "
            "Growth figures for other quarters are not provided in the source material.'\n\n"
            "Example 2 — Related but not directly answering:\n"
            "Context: 'The company hired 200 engineers in 2024.'\n"
            "Question: 'What is the total headcount?'\n"
            "Answer: 'The context mentions 200 engineers were hired in 2024, but does not "
            "state the total headcount. I cannot determine this from the available information.'\n\n"
            "Example 3 — Completely unanswerable:\n"
            "Context: 'The product launched in March 2024.'\n"
            "Question: 'What is the CEO salary?'\n"
            "Answer: 'This information is not present in the provided context.'"
            "\n\nCitation requirement: For every factual claim, cite the source using "
            "the format [Source: filename, p.N]. The source metadata is provided in "
            "each chunk header. Example: 'Revenue grew 15% [Source: report.pdf, p.3].'"
        )),
        HumanMessage(content=f"{history_text}Context:\n{context}\n\nQuestion: {question}"),
    ]
    try:
        answer = model.invoke(messages).content
    except Exception as e:
        print(f"   ❌ Answer generation failed: {e}")
        return AgentResult(status="error", error=str(e), is_retryable=True, metadata={"agent": "answer"})
    print(f"   Initial answer drafted.")
    return AgentResult(status="success", data=answer, metadata={"agent": "answer"})


# ─── Agent 4: Critic ──────────────────────────────────────────────────────
def critic_agent(question: str, chunks: list, answer: str) -> AgentResult:
    print(f"\n🧐 Critic Agent: Reviewing answer for gaps/inaccuracies...")
    context = build_context(chunks)
    messages = [
        SystemMessage(content=(
            "You are a strict fact-checker. Compare the answer to the source chunks. "
            "Check for: (1) gaps — important info in chunks missing from the answer, "
            "(2) unsupported claims — statements not backed by any chunk, "
            "(3) fabricated details — names, dates, or numbers not present in the source, "
            "(4) missing citations — claims without [Source:] references. "
            "If the answer is fully correct and complete, say 'NO ISSUES'. "
            "Otherwise, provide a short bulleted critique with specific issues."
        )),
        HumanMessage(content=(
            f"Question: {question}\n\nSource Chunks:\n{context}\n\n"
            f"Answer to review:\n{answer}"
        )),
    ]
    try:
        critique = model.invoke(messages).content
    except Exception as e:
        print(f"   ❌ Critique failed: {e}")
        return AgentResult(status="error", error=str(e), is_retryable=True, metadata={"agent": "critic"})
    print(f"   Critique complete.")
    return AgentResult(status="success", data=critique, metadata={"agent": "critic"})


# ─── Agent 5: Refiner ─────────────────────────────────────────────────────
def refiner_agent(question: str, chunks: list, answer: str, critique: str, history: list) -> AgentResult:
    print(f"\n✨ Refiner Agent: Producing final answer...")
    context = build_context(chunks)
    history_text = ""
    if history:
        history_text = "Conversation History:\n" + "\n".join(
            f"Q: {h['question']}\nA: {h['answer']}" for h in history
        ) + "\n\n"
    messages = [
        SystemMessage(content=(
            "You produce a final, high-accuracy answer grounded in the source chunks. "
            "Incorporate the critic's feedback to fix gaps or inaccuracies. "
            "Stay strictly faithful to the context. "
            "Use the conversation history for continuity with prior questions."
            " Preserve all source citations from the initial answer. Add citations "
            "for any new claims. Use format [Source: filename, p.N]."
        )),
        HumanMessage(content=(
            f"{history_text}Question: {question}\n\nSource Chunks:\n{context}\n\n"
            f"Initial Answer:\n{answer}\n\nCritic Feedback:\n{critique}\n\n"
            "Write the final refined answer:"
        )),
    ]
    try:
        refined = model.invoke(messages).content
    except Exception as e:
        print(f"   ❌ Refinement failed: {e}")
        return AgentResult(status="error", error=str(e), is_retryable=True, metadata={"agent": "refiner"})
    print(f"   Final answer ready.")
    return AgentResult(status="success", data=refined, metadata={"agent": "refiner"})


# ─── Agent 6: Orchestrator ────────────────────────────────────────────────
def orchestrator(vectorstore: Chroma, question: str, history: list) -> AgentResult:
    print(f"\n{'='*60}")
    print(f"🧠 Orchestrator: Pipeline for: {question!r}")
    print(f"{'='*60}")

    retrieval = retrieval_agent(vectorstore, question)

    if retrieval.status == "error":
        print(f"\n{'='*60}")
        print(f"Orchestrator: Pipeline aborted — retrieval error.")
        print(f"{'='*60}")
        return retrieval

    if retrieval.status == "no_results":
        msg = f"I couldn't find any relevant information in the loaded PDFs to answer: \"{question}\""
        print(f"\n{'='*60}")
        print(f"Orchestrator: No relevant chunks — skipping answer pipeline.")
        print(f"{'='*60}")
        return AgentResult(
            status="no_results",
            data=msg,
            metadata={"agent": "orchestrator", "reason": "retrieval_below_threshold"},
        )

    chunks = retrieval.data

    initial = answer_agent(question, chunks, history)
    if initial.status == "error":
        return initial

    current_answer = initial.data
    retries = 0

    for attempt in range(MAX_REFINE_RETRIES + 1):
        critique = critic_agent(question, chunks, current_answer)
        if critique.status == "error":
            result = AgentResult(status="success", data=current_answer, metadata={"agent": "orchestrator"})
            result.metadata["degraded"] = True
            result.metadata["failed_stage"] = "critic"
            result.metadata["retries"] = retries
            return result

        if "NO ISSUES" in critique.data.upper():
            print(f"   Critic approved on attempt {attempt + 1}.")
            break

        print(f"   🔄 Critic found issues (attempt {attempt + 1}/{MAX_REFINE_RETRIES + 1}), retrying refinement...")
        refined = refiner_agent(question, chunks, current_answer, critique.data, history)
        if refined.status == "error":
            result = AgentResult(status="success", data=current_answer, metadata={"agent": "orchestrator"})
            result.metadata["degraded"] = True
            result.metadata["failed_stage"] = "refiner"
            result.metadata["retries"] = retries
            return result
        current_answer = refined.data
        retries += 1

    print(f"\n{'='*60}")
    print("Orchestrator: Pipeline complete.")
    print(f"{'='*60}")
    return AgentResult(
        status="success",
        data=current_answer,
        metadata={"agent": "orchestrator", "retries": min(retries, MAX_REFINE_RETRIES)},
    )


# ─── CLI ──────────────────────────────────────────────────────────────────
def main():
    vectorstore = ingestion_agent()
    history = []

    print("\nReady. Type a question (or 'quit' to exit).")
    print("Type 'clear history' to reset conversation context.")
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
        if q.lower() == "clear history":
            history.clear()
            print("🗑️  Conversation history cleared.")
            continue
        result = orchestrator(vectorstore, q, history)

        if result.status == "error":
            print(f"\n⚠️  Error: {result.error}")
            if result.is_retryable:
                print("   This may be a temporary issue. Try again.")
            continue

        if result.status == "no_results":
            print(f"\n📄 {result.data}\n")
            continue

        history.append({"question": q, "answer": result.data})
        print(f"\n📄 FINAL ANSWER:\n{result.data}\n")

    print("Goodbye.")


if __name__ == "__main__":
    main()
