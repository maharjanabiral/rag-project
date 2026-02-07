from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class ChunkingService:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_documents(self, path: str) -> List[Document]:
        path_obj = Path(path)
        loader = DirectoryLoader(str(path_obj), glob="**/*.pdf", loader_cls=PyMuPDFLoader) if path_obj.is_dir() else PyMuPDFLoader(str(path_obj))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = Path(doc.metadata.get("source", "unknown")).name
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)