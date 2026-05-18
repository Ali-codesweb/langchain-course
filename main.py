from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()


@tool
def search(query: str) -> str:
    """
    This tool searches from the internet.
    Args:
    - Query (str): The query to search for.
    Results:
    - The search result
    """
    print(f"Searching for {query}...")
    return "Tokyo weather is fine"


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather like in Tokyo?")]}
    )
    print(response)


if __name__ == "__main__":
    main()
