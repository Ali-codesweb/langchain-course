from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_classic.tools import tool
from langchain_community.tools import tool
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchTool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

ddg_search = DuckDuckGoSearchTool()


@tool
def get_search_result(query: str):
    """
    This function searches from the internet

    Args:
        text (str): the query to search

    Returns:
        result (str): The result of the search
    """
    result = ddg_search.invoke(query)
    return result


def main():

    tools = [get_search_result]
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
    )

    agent = create_react_agent(llm=llm, tools=tools)

    agent_executor = AgentExecutor(agent=agent, tools=tools)


if __name__ == "__main__":
    main()
