from modules.chunking import ChunkingService
from modules.embeddings import EmbeddingService
from modules.rag_service import RAGService

class RAGPipeline:
    def __init__(self):
        self.chunker = ChunkingService()
        self.embedder = EmbeddingService()
        self.db = self.embedder.get_db()

    def list_indexed_files(self):
        data = self.db.get()
        if not data or not data['metadatas']: return []
        return list(set(m['source_file'] for m in data['metadatas']))
    
    def ingest(self, path: str):
        # No more clear_db()! We keep everything.
        docs = self.chunker.load_documents(path)
        chunks = self.chunker.split_documents(docs)
        self.embedder.add_documents(self.db, chunks)

    async def query(self, question: str):
        # 1. Get list of all filenames currently in DB
        filenames = self.list_indexed_files()
        
        # 2. Let the RAG Service handle the routing internally
        rag_instance = RAGService(self.db)
        return await rag_instance.answer(question, filenames)

    
