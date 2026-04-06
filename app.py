import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

if len(sys.argv) < 2:
    print("Usage: python app.py <path-to-pdf>")
    sys.exit(1)

pdf_path = sys.argv[1]
print(f"Loading PDF: {pdf_path}")

# Load PDF
docs = PyPDFLoader(pdf_path).load()
print(f"Loaded {len(docs)} pages")

# Split into chunks
chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
).split_documents(docs)
print(f"Created {len(chunks)} chunks")

# Embed and store in ChromaDB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
print("Embeddings stored in ChromaDB")

# Build QA chain using modern LangChain API
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_template(
    "Answer the question based only on the following context:\n\n"
    "{context}\n\n"
    "Question: {question}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\nReady! Ask questions about your PDF (type 'quit' to exit).\n")

while True:
    try:
        question = input("Question: ")
        if question.lower() in ("quit", "exit", ""):
            print("Goodbye!")
            break
        answer = chain.invoke(question)
        print(f"\nAnswer: {answer}\n")
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
