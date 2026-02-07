import asyncio
from app.services.rag_pipeline import RAGPipeline

async def main():
    # Initialize pipeline
    pipeline = RAGPipeline(source_dir="./source", persist_dir="./db/chroma")

    # Step 1: Load existing embeddings or ingest PDFs
    pipeline.ingest_or_load_db()

    # Step 2: Query RAG system
    question = "What is there in table 19 of this document?"
    answer = await pipeline.query(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    asyncio.run(main())
