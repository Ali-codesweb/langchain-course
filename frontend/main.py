import asyncio
import os
from typing import Any, List

import streamlit as st

from backend.core import run_llm


def _format_sources(context_docs: List[Any]) -> List[str]:
    """Extract and format unique source URLs/paths from context chunks."""
    sources = set()  # Using a set to prevent duplicate URLs in the expander
    for doc in context_docs:
        meta = getattr(doc, "metadata", None) or {}
        # Checks for 'url' first (from web crawl), falls back to 'source' (from local files)
        source_val = meta.get("url") or meta.get("source")
        if source_val:
            sources.add(str(source_val))
    return list(sources)


# Set page configuration at the absolute top-level
st.set_page_config(page_title="Langchain Documentation Assistant", page_icon="📚")


async def main():
    st.title("Langchain Documentation Assistant")

    # Sidebar session management
    with st.sidebar:
        st.subheader("Session")
        if st.button("Clear chat", use_container_width=True):
            st.session_state.pop("messages", None)
            st.rerun()

    # Initialize session state with welcome message
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! I'm your Langchain documentation assistant. How can I help you?",
                "sources": [],
            }
        ]

    # Render conversation history from state memory
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Only render the sources expander if sources exist
            if msg.get("sources"):
                with st.expander("Sources:"):
                    for source in msg["sources"]:
                        st.markdown(f"- {source}")

    # Capture user input
    prompt = st.chat_input("Ask anything about Langchain...")

    if prompt:
        # Append and render user message
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "sources": []}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        # Target assistant context window
        with st.chat_message("assistant"):
            try:
                # Execution wrapper inside the loading spinner
                with st.spinner("Thinking..."):
                    # ✅ Now perfectly legal because it executes inside 'async def main()'
                    response = run_llm(prompt)
                    # If your run_llm backend function is 'async def', keep the await prefix:
                    if asyncio.iscoroutine(response):
                        response = await response

                # Extract text answer and format unique sources array
                answer_text = response["answer"]
                formatted_sources = _format_sources(response.get("context", []))

                # Render components to the screen immediately
                st.markdown(answer_text)
                if formatted_sources:
                    with st.expander("Sources:"):
                        for source in formatted_sources:
                            st.markdown(f"- {source}")

                # Append finalized package back into state tracker history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "sources": formatted_sources,
                    }
                )

            except Exception as e:
                st.error("Failed to get response")
                st.exception(e)


# Launch the stream event loop wrapper
# Replace your current if __name__ == "__main__": block with this:
if __name__ == "__main__":
    try:
        # 1. Attempt to fetch an already running loop context if it exists
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # 2. If no loop is active, fetch or create a standard event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        # 3. Maintain the loop context safely across Streamlit thread refreshes
        loop.run_until_complete(main())
    except Exception as e:
        st.error(f"Event loop exception occurred: {e}")
