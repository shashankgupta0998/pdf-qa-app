
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

search = DuckDuckGoSearchRun()

# ─── Agent 1: Researcher ───────────────────────────────────────────────────


def researcher_agent(topic: str) -> str:
    print(f"\n🔍 Researcher Agent: Searching for '{topic}'...")

    # Search twice for comprehensive results
    results1 = search.run(f"{topic} 2025")
    results2 = search.run(f"{topic} latest developments")

    combined = f"Search 1: {results1}\n\nSearch 2: {results2}"

    # Summarize the search results
    messages = [
        SystemMessage(content="You are a research assistant. Summarize the key facts from the search results provided. Be concise and factual. Extract only the most important points."),
        HumanMessage(
            content=f"Summarize these search results about {topic}:\n\n{combined}")
    ]

    summary = model.invoke(messages)
    print(f"   Research complete.")
    return summary.content

# ─── Agent 2: Writer ──────────────────────────────────────────────────────


def writer_agent(topic: str, research: str) -> str:
    print(f"\n✍️  Writer Agent: Writing report on '{topic}'...")

    messages = [
        SystemMessage(content="""You are a professional report writer. 
Write a clear, structured report based on the research provided.
Format:
- Title
- Executive Summary (2-3 sentences)
- Key Findings (3-5 bullet points)
- Conclusion (2-3 sentences)
Keep it concise and professional."""),
        HumanMessage(
            content=f"Write a report about {topic} based on this research:\n\n{research}")
    ]

    report = model.invoke(messages)
    print(f"   Report written.")
    return report.content

# ─── Orchestrator ─────────────────────────────────────────────────────────


def orchestrator(topic: str) -> str:
    print(f"\n{'='*50}")
    print(f"Orchestrator: Starting research pipeline for '{topic}'")
    print(f"{'='*50}")

    # Step 1 — Run researcher agent
    research = researcher_agent(topic)

    # Step 2 — Pass research to writer agent
    report = writer_agent(topic, research)

    print(f"\n{'='*50}")
    print("Orchestrator: Pipeline complete.")
    print(f"{'='*50}")

    return report


# ─── Run it ───────────────────────────────────────────────────────────────
final_report = orchestrator("Model Context Protocol in AI development")
print(f"\n📄 FINAL REPORT:\n")
print(final_report)
