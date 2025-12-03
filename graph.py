# graph.py - Statitstical Learning Agent

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv, dotenv_values
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
load_dotenv()


PDF_PATH = "./the Elements of Statistical Learning.pdf"


# Define State Schema
class RAGState(TypedDict):
    """State for statistical learning agent workflow"""
    query: str
    answer: str
    source: str

    

# Load and Build Vectorstore
def load_vectorstore():
    print("Load PDF and built vectorbase...")

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    # chunk PDF pages
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embedding)
    return vectorstore


VECTORSTORE = load_vectorstore()


llm = ChatOpenAI(model="gpt-4o", temperature=0.1)



# Node: Query
def query_node(state: RAGState):
    query = state["query"]

    # similarity search with metadata
    docs = VECTORSTORE.similarity_search(query, k=4)

    # Build context with source information
    context_parts = []
    source_info = []

    for i, doc in enumerate(docs, 1):
        # Extract page number from metadata
        page_num = doc.metadata.get('page', 'Unknown')
        source_file = doc.metadata.get('source', 'Unknown')

        # Add to context with reference number
        context_parts.append(f"[Source {i}] (Page {page_num}):\n{doc.page_content}\n")

        # Store source information
        source_info.append({
            'source_num': i,
            'page': page_num,
            'excerpt': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        })

    context = "\n".join(context_parts)

    prompt = f"""
    You are a RAG assistant. You must answer ONLY using the information in the provided sources.
    When referencing information, cite the source number (e.g., [Source 1]).
    If the answer is not directly stated in the sources, respond:
    "No related information found."

    Do NOT infer, deduce, or guess.
    Do NOT use any prior knowledge you may have.

    PDF Content with Sources:
    {context}

    Question:
    {query}

    Please provide your answer with citations.
    """

    answer = llm.invoke(prompt).content

    # Format source references
    source_text = "\n\n## Sources:\n"
    for src in source_info:
        source_text += f"\n**Source {src['source_num']}** (Page {src['page']}):\n"
        source_text += f"Excerpt: {src['excerpt']}\n"

    state["answer"] = answer
    state["source"] = source_text
    return state

# Build the Graph

graph = StateGraph(RAGState)

# Add nodes
graph.add_node("query_node", query_node)

# Add edges
graph.add_edge(START, "query_node")
graph.add_edge("query_node", END)


graph = graph.compile()

app = graph




