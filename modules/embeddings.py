from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "./db/chroma"):
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        self.persist_directory = persist_directory

    def get_db(self) -> Chroma:
        return Chroma(
            collection_name="rag_collection", 
            embedding_function=self.embedding_model, 
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space" : "cosine"}
        )

    def add_documents(self, db: Chroma, documents: List[Document]):
        """
        Adds documents only if the file (source_file) is not already in the DB.
        """
        if not documents:
            print("No documents provided.")
            return

        incoming_sources = {
            doc.metadata.get("source_file") 
            for doc in documents 
            if doc.metadata.get("source_file")
        }

        if not incoming_sources:
            print("Warning: Documents missing 'source_file' metadata. Skipping deduplication logic.")
            db.add_documents(documents)
            return

        try:
            existing_records = db.get(
                where={"source_file": {"$in": list(incoming_sources)}}
            )

            existing_sources = {m["source_file"] for m in existing_records.get("metadatas", [])}
        except Exception as e:
            print(f"Note: Could not query existing sources (DB might be empty): {e}")
            existing_sources = set()

        to_add = [
            doc for doc in documents 
            if doc.metadata.get("source_file") not in existing_sources
        ]

        if to_add:
            new_files_count = len({d.metadata["source_file"] for d in to_add})
            print(f"Found {new_files_count} new file(s). Adding {len(to_add)} chunks...")
            db.add_documents(to_add)
        else:
            print("All files already exist in the database. Skipping embedding.")

    def delete_source_file(self, db: Chroma, source_file: str):
        """
        Helper method: If you ever want to re-index a specific file, 
        call this first to remove its old chunks.
        """
        existing = db.get(where={"source_file": source_file})
        if existing["ids"]:
            db.delete(ids=existing["ids"])
            print(f"Deleted existing chunks for {source_file}")