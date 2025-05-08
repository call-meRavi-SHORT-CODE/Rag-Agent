from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model # to load groq model
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.tools import DuckDuckGoSearchResults  # Free web search
import os
import requests
from typing import Any, Dict
from typing import List, Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model # to load groq model
from langgraph.prebuilt import create_react_agent
from langchain_ollama.llms import OllamaLLM






INDEX_DIR = "faiss_index"
TAVILY_API_KEY="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"
os.environ.setdefault("USER_AGENT", "Doc_Search_Agent")  
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# Embedding from Hugging Face
hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists(INDEX_DIR):
    #  loading the Vector Store if already exist 
    db = FAISS.load_local(
            INDEX_DIR,
            hf_embeddings,
            allow_dangerous_deserialization=True)
else:
    # Data Ingestion, Transforming into Chunks, Embeddings

    loader = WebBaseLoader(
        web_paths=("https://docs.docker.com/get-started/docker-overview/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(class_=("flex w-full gap-8",))
        ),
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    db = FAISS.from_documents(split_docs, hf_embeddings)

    # Saving the Vector Store
    db.save_local(INDEX_DIR)


# Using Llama LLM using OLlama

llm = OllamaLLM(model="llama3.1")




# Designing ChatPrompt Template

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
<context>
{context}
</context>
Question: {input}""")


# Output Praser
parser = StrOutputParser()


# Creating a Custom Chain 
document_chain = create_stuff_documents_chain(
    llm,
    prompt,
    output_parser=parser,
)

# Retriver
retriever=db.as_retriever()

qa_chain = create_retrieval_chain(retriever, document_chain)


## Tool1 

@tool
def doc_qa(query: str) -> str:
    """Answer questions based on the Docker documentation context."""
    result: Dict[str, Any] = qa_chain.invoke({"input": query})
    return result.get('answer', 'No answer found.')


@tool
def search_web(query: str) -> list:
    """Perform a free web search using DuckDuckGo."""
    ddg = DuckDuckGoSearchResults()
    return ddg.run(query)



tools = [doc_qa, search_web]

system_prompt = """
You are an AI assistant with access to the following tools:

- doc_qa: answers questions about Docker documentation using a FAISS-based retrieval chain.
- search_web: performs a web search when doc_qa does not provide a relevant answer.

Behavior:
1. If the user asks about Docker usage or documentation, always try doc_qa first.
2. If doc_qa yields no relevant answer, use search_web to find and return the best possible response.
3. If neither tool can address the query, reply based on your own knowledge.
"""

agent = create_react_agent(model=llm, tools=tools, state_modifier=system_prompt)


def run_agent(question: str):
    inputs = {"messages": [("user", question)]}
    for step in agent.stream(inputs, stream_mode="values"):
        msg = step['messages'][-1]
        msg.pretty_print()

# Example usage
if __name__ == "__main__":
    run_agent("How to use a Docker Compose to run a multi-container application?")
