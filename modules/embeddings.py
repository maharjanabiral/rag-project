from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "../db/chroma"):
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)

    def create_db(self, documents: List[Document], collection_name: str = "pdf_chunks") -> Chroma:
        """
        Create and persist a Chroma vector store from a list of Document objects.
        """
        db = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            collection_name=collection_name,
            persist_directory=self.persist_directory    
        )
        db.persist()
        print(f"Chroma DB created at {self.persist_directory} with {len(documents)} documents")
        return db

    def load_db(self, collection_name: str = "pdf_chunks") -> Chroma:
        """
        Load an existing Chroma vector store.
        """
        db = Chroma(
            persist_directory=self.persist_directory,
            collection_name=collection_name,
            embedding_function=self.embedding_model
        )
        print(f"Chroma DB loaded from {self.persist_directory}")
        return db
