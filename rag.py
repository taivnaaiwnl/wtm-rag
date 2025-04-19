import os
from typing import List

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders.text import TextLoader  # noqa: F811
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

os.environ["GOOGLE_API_KEY"] = "AIzaSyDKLrXSPRj8zXg6vSCZu-EqaHQXoOCWWbM"

def load_documents(directory_path: str):
    loader = DirectoryLoader(
        directory_path, 
        glob="**/*.txt",
        loader_cls=lambda file_path: TextLoader(file_path, encoding="utf-8")
    )
    documents = loader.load()
    return documents

def setup_text_splitters():
    separators = ["\n\n", "\n", ". ", ", ", " ", ""]
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=separators,
        is_separator_regex=False
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=separators,
        is_separator_regex=False
    )
    return parent_splitter, child_splitter

def setup_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(
        collection_name="rag_documents",
        embedding_function=embeddings
    )
    docstore = InMemoryStore()
    return embeddings, vectorstore, docstore

def create_retriever(
    documents: List, 
    vectorstore, 
    docstore, 
    parent_splitter, 
    child_splitter, 
    k: int = 4
):
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        k=k
    )
    retriever.add_documents(documents)
    return retriever

def setup_gemini_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        top_p=0.85,
        top_k=40,
        max_output_tokens=2048,
    )
    return llm

def create_rag_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_template("""
        You are a highly trained virtual assistant specialized in providing accurate, professional support for company-information-related queries.
    Use only the information provided in the context to answer questions. Do not reference or infer details from external sources.
    If the necessary information is absent from the context, politely inform the user that you do not know or that the information is unavailable.
    Keep responses concise and relevant, ensuring clarity in a maximum of five sentences.
    Always maintain a polite, professional, and helpful tone, focusing on delivering precise, clear, and customer-centric answers.
    For questions involving specific terms, phrases, or data in Latin, convert them into Cyrillic and interpret them using the context to provide the most accurate response.
    Respond in the same language as the user's inquiry to ensure a seamless communication experience.
    When the user greets you (e.g., "hi," "hello," "сайн уу," "сайн байна уу"), respond appropriately with a polite greeting, using the same language.

    Always list the sources exactly as they appear in the context provided.
    
    Context: {context}
    Question: {question}

    Answer: [Provide your concise answer here based on the context.]
    Sources: [List only the sources from the context used for your answer.]
    """)
    output_parser = StrOutputParser()
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )
    return rag_chain

def setup_rag_system(directory_path: str, top_k: int = 4):
    documents = load_documents(directory_path)
    parent_splitter, child_splitter = setup_text_splitters()
    embeddings, vectorstore, docstore = setup_vectorstore()
    retriever = create_retriever(
        documents, 
        vectorstore, 
        docstore, 
        parent_splitter, 
        child_splitter, 
        k=top_k
    )
    llm = setup_gemini_llm()
    rag_chain = create_rag_chain(retriever, llm)
    return rag_chain

def normalize_cyrillic_text(text):
    import unicodedata
    return unicodedata.normalize('NFKC', text)

if __name__ == "__main__":
    DOCS_DIR = "C:/Users/taivn/coding/wtm-rag"
    rag_chain = setup_rag_system(DOCS_DIR, top_k=5)
    queries = [
        "Hoolnii mungu hed ve", 
        'eeljiin amralt hed ve',
        'huuhed maani turchihluu yaah ve'
    ]
    for query in queries:
        print(f"\nАсуулт (Query): {query}")
        response = rag_chain.invoke(query)
        print(f"Хариулт (Response): {response}")
