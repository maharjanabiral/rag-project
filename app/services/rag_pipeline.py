import asyncio
from modules.chunking import ChunkingService
from modules.embeddings import EmbeddingService
from app.services.rag import RAGService
from pathlib import Path

class RAGPipeline:
    def __init__(self, source_dir: str="../source", persist_dir: str="../db/chroma"):
        self.source_dir = source_dir
        self.persist_dir = persist_dir
        self.chunk_service = ChunkingService()
        self.embedding_service = EmbeddingService(persist_directory=self.persist_dir)

    def ingest_or_load_db(self):
        """Load existing embeddings or ingest PDFs and create embeddings."""
        if Path(self.persist_dir).exists() and any(Path(self.persist_dir).iterdir()):
            print(f"Loading existing Chroma DB from {self.persist_dir}")
            self.db = self.embedding_service.load_db()

        else:
            documents = self.chunk_service.load_documents(self.source_dir)
            chunks = self.chunk_service.split_documents(documents)
            self.db = self.embedding_service.create_db(chunks)
        
        self.rag_service = RAGService(self.db)

    async def query(self, question: str) -> str:
        if self.rag_service is None:
            raise ValueError("DB not loaded. Call ingest_or_load_db() first.")
        return await self.rag_service.answer(question)
        