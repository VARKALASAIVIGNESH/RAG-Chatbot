import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
load_dotenv()

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

def load_documents(file_path="data/knowledge_base.txt"):
    loader = TextLoader(file_path)
    documents = loader.load()  # This is still a list of 1 Document
    
    text_splitter = CharacterTextSplitter(
        separator="\n\n",   # split on blank lines (paragraphs)
        chunk_size=500,     # max characters per chunk
        chunk_overlap=50    # overlap chars for context between chunks
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Loaded {len(chunks)} documents after splitting")
    if chunks:
        print(f"First chunk content (sample): {chunks[0].page_content[:200]}")
    return chunks


def get_vectorstore(documents=None, persist_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(persist_path):
        vectorstore = FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
    else:
        if not documents:
            raise ValueError("No documents loaded for creating vectorstore.")
        texts = [doc.page_content for doc in documents if isinstance(doc, Document)]
        print(f"Prepared {len(texts)} texts for embedding")
        vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
        vectorstore.save_local(persist_path)
    return vectorstore

def get_rag_chain():
    documents = load_documents()
    vectorstore = get_vectorstore(documents)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatOpenAI(
        openai_api_key=os.getenv("GROQ_API_KEY"),
        openai_api_base=os.getenv("GROQ_API_BASE"),
        model=os.getenv("GROQ_MODEL"),
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )
    return qa_chain

def get_rag_response(query: str):
    chain = get_rag_chain()
    return chain.run(query)
