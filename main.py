from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from typing import List
from pydantic import BaseModel, Field


class Source(BaseModel):
    """
    Scheme for a source used by the agent
    """

    url: str = Field(..., description="URL of the source")


class AgentResponse(BaseModel):
    """
    Scheme for the agent response
    """

    answer: str = Field(..., description="Answer to the query")
    sources: List[Source] = Field(..., description="Sources used to answer the query")


load_dotenv()

# Rename the actual search instance
duckduckgo_search = DuckDuckGoSearchRun()


@tool
def internet_search(query: str) -> str:
    """
    Search the internet using DuckDuckGo.

    Args:
        query (str): Search query

    Returns:
        str: Search results
    """
    print(f"Searching for: {query}")
    results = duckduckgo_search.invoke(query)
    return results


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

tools = [internet_search]

agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(content="search me 3 active jobs on linkedin for kuwait?")
            ]
        }
    )

    print(response)


if __name__ == "__main__":
    main()
