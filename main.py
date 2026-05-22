import os
from operator import itemgetter
from typing import Any

from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_postgres import PGVector

load_dotenv()

print("Initializing components")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
# llm = ChatOllama(model="qwen3.5:4b")
llm = ChatOpenAI(
    base_url="http://127.0.0.1:8111/v1",
    api_key="trtrtrtr",
    model="mlx-community/Qwen3-4B-Instruct-2507-4bit",
)
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="documents",
    connection=os.environ.get("PGVECTOR_DB_URL"),
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
prompt_template = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant that answers questions based on the context provided below. Answer in a clear, concise, and informative manner.
    
    Context:\n\n{context}\n\n
    Question: {question}\n\n
    Provide a Detailed answer:
    """
)


def format_docs(docs):
    """Format retreived docs"""
    return "\n\n".join(document.page_content for document in docs)


def retrieval_chain_without_lcel(query: str):
    "Old"

    # retreive the documents
    docs = retriever.invoke(query)

    # format the documents
    context = format_docs(docs)

    # format the prompt
    prompt = prompt_template.format_messages(context=context, question=query)

    # invoke the model
    response = llm.invoke(prompt)

    return response


def retrieval_chain_with_lcel() -> RunnableSerializable[dict[str, Any], str]:
    """New"""

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


if __name__ == "__main__":
    print("retreiving...")
    query = "What is PGVector in machine learning?"
    # result = retrieval_chain_without_lcel(query)
    result1 = retrieval_chain_with_lcel()

    # print(result.content)
    print(result1.invoke({"question": query}))
