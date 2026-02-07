from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from typing import List

class ChunkingService:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
    def load_documents(self, docs_path="../source"):
        print(f"Loading documents from {docs_path}")

        loader = DirectoryLoader(
            path=docs_path,
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader
        )

        documents = loader.load()
        print(f"Loaded {len(documents)} pages")

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)

        for i, chunk in enumerate(chunks[:5], start=1):
            print(f"Chunk {i}: ")
            print(f"Chunk content: {chunk.page_content}")
            print(f"Chunk metadata: {chunk.metadata}")

        return chunks

# For Easy Testing
if __name__ == "__main__":
    service = ChunkingService(
        chunk_size=600,
        chunk_overlap=50
    )

    documents = service.load_documents(docs_path="../source")
    chunks = service.split_documents(documents=documents)