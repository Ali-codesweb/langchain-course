from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun

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

agent = create_agent(
    model=llm,
    tools=tools,
)


def main():
    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather like in Tokyo?")]}
    )

    print(response)


if __name__ == "__main__":
    main()
