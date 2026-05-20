from dotenv import load_dotenv
from langchain_classic.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama

load_dotenv()


@tool
def get_text_length(text: str):
    """
    This function counts the number of words in a text

    Args:
        text (str): the text to count

    Returns:
        result (int): The number of words in the text
    """
    length = len(text.split())
    return length


def main():

    tools = [get_text_length]
    llm = ChatOllama(
        model="qwen2.5-coder:3b",
    )

    # Bind tools directly to the LLM
    llm_with_tools = llm.bind_tools(tools)

    # Initial human message
    messages = [HumanMessage(content="What is the length of the word 'DOG'?")]

    while True:
        # First turn: LLM decides which tools to call
        ai_msg = llm_with_tools.invoke(messages)
        tool_calls = ai_msg.tool_calls

        if len(tool_calls) > 0:
            messages.append(ai_msg)
            # Second turn: Execute tool calls and pass results back
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")

                # Find the matching tool
                tool_map = {tool.name: tool for tool in tools}
                selected_tool = tool_map[tool_name]

                # Execute the tool
                observation = selected_tool.invoke(tool_args)

                # Append the tool execution result
                messages.append(
                    ToolMessage(
                        content=str(observation),
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                )
            continue

        response = llm_with_tools.invoke(messages)
        print(response.content)
        break


if __name__ == "__main__":
    main()
