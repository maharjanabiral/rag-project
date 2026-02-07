from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

class ChunkingService:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents=documents)
