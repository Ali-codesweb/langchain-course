import asyncio
import os
from typing import Dict

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from sqlalchemy.dialects.postgresql import Any

from config.settings import vectorstore

load_dotenv()
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool

#  base_url="http://127.0.0.1:8111/v1",
# api_key="trtrtrtr",
# model="mlx-community/Qwen3-4B-Instruct-2507-4bit",

model = init_chat_model(
    model="mlx-community/Qwen3-4B-Instruct-2507-4bit",
    model_provider="openai",  # 👈 This resolves the ValueError
    base_url="http://127.0.0.1:8111/v1",
    api_key="trtrtrtr",
    # model_provider="google_genai",
    # model="gemini-2.5-flash",
)


@tool(response_format="content_and_artifact")
async def retreive_context(query: str):
    """Retreive relevant documentation to help answer user queries about Langchain"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieved_documents = await retriever.ainvoke(query)

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('url', 'Unknown')}\n\n Content: {doc.page_content}"
        for doc in retrieved_documents
    )

    return serialized, retrieved_documents


async def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    Args:
        query (str): The user's query
    Returns:
        Dict[str,Any]: The response from the LLM with the list of retrieved documents.
    """

    # This prompt provides a strict behavioral framework that smaller models understand
    system_prompt = (
        "You are a specialized LangChain Documentation assistant.\n\n"
        "STEP 1 - FACT-GATHERING PHASE:\n"
        "Before writing any narrative answer or sentence, you MUST execute the `retreive_context` tool. "
        "Do not guess. Do not reply using your own internal knowledge. You must call the tool first to look up files.\n\n"
        "STEP 2 - ANSWERING PHASE:\n"
        "Once you receive the tool's output context, read it thoroughly and formulate your final response to the user. "
        "Always synthesize your final answer using ONLY the factual content found within the tool's response."
    )
    agent = create_agent(
        model=model,
        tools=[retreive_context],
        system_prompt=system_prompt,
    )

    messages = [{"role": "user", "content": query}]
    response = await agent.ainvoke({"messages": messages})
    answer = response["messages"][-1].content

    print(response)
    context_docs = []

    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {
        "answer": answer,
        "context": context_docs,
    }


if __name__ == "__main__":
    asyncio.run(run_llm("What are deep agents?"))
