import os

from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool


model =init_chat_model(model=)