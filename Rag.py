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
import os



INDEX_DIR = "faiss_index"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ.setdefault("USER_AGENT", "Doc_search_Agent")  # Set a default user agent to avoid warning


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


# Using Groq LLM 

llm = init_chat_model("llama3-8b-8192", model_provider="groq")


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

# combining Chain|Retriver as Retriver chain 
retrieval_chain=create_retrieval_chain(retriever,document_chain)

response=retrieval_chain.invoke({"input":"how to use a Docker Compose to run a multi-container application "})
a = response['answer']
print(a)