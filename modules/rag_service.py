import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

class RAGService:
    def __init__(self, db):
        self.db = db
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=os.getenv("GROQ_API_KEY"))
        
        # 1. Router Prompt: Decides which file to look at
        self.router_prompt = ChatPromptTemplate.from_template("""
            You are an expert router. Given a list of filenames and a user question, 
            pick the most relevant filename to answer the question.
            
            Filenames: {filenames}
            Question: {question}
            
            Return ONLY the filename. If no file is relevant, return "None".
        """)
        self.router_chain = self.router_prompt | self.llm | StrOutputParser()

    async def get_relevant_file(self, question: str, filenames: list) -> str:
        """Determines which file metadata to filter by."""
        if not filenames: return None
        response = await self.router_chain.ainvoke({"filenames": filenames, "question": question})
        selected = response.strip()
        return selected if selected in filenames else None

    def _get_retriever(self, filename: str):
        """Creates a filtered retriever on the fly."""
        search_kwargs = {"k": 3}
        if filename:
            search_kwargs["filter"] = {"source_file": filename}
            
        return self.db.as_retriever(
            search_kwargs=search_kwargs
        )

    async def answer(self, question: str, filenames: list):
        # Step 1: Automatically find the right file
        selected_file = await self.get_relevant_file(question, filenames)
        
        # Step 2: Set up a chain with that specific file filter
        retriever = self._get_retriever(selected_file)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are answering from the document: {selected_file if selected_file else 'Knowledge Base'}."),
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])

        chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt | self.llm | StrOutputParser()
        )
        
        return await chain.ainvoke(question)

    def _format_docs(self, docs):
        return "\n\n".join([d.page_content for d in docs])