from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv

load_dotenv()

class RAGService:
    def __init__(self, db):
        self.retriever = db.as_retriever(search_kwargs={"k":3})
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )

        self.template = """
        You are a professional assistant. Use the following context to answer the question.
        If you don't know the answer, say you don't know. 
        
        Context: {context}
        Question: {question}
        Answer:"""
        self.prompt = ChatPromptTemplate.from_template(self.template)

    async def answer(self, question: str):
        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()} | self.prompt | self.llm | StrOutputParser()
        )

        return await chain.ainvoke(question)