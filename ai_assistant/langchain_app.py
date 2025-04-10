from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.llms import Ollama
from langchain.tools import BaseTool
from typing import Optional, Type
import requests
from tools import calculator_tool, search_tool, chat_tool

import os
from dotenv import load_dotenv
LAAMA_3B = os.getenv("LAAMA_3B")
OLLAMA_URL = os.getenv("OLLAMA_URL")

# --- Define tools for LangChain ---
tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Useful for solving math problems. Input should be a math expression like '2 * 6'"
    ),
    Tool(
        name="Search",
        func=search_tool,
        description="Useful for looking up current or recent information like 'news about Tata Group'"
    ),
    Tool(
        name="Chat",
        func=chat_tool,
        description="Use this for casual greetings or social conversation like 'hello', 'how are you', etc."
    )
]

# --- Initialize Ollama LLM ---
llm = Ollama(
    model=LAAMA_3B,  # You can change to "gemma3:12b" etc.
    base_url=OLLAMA_URL,  # Use your IP here
    temperature=0.7
)

# --- Create agent ---
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    max_iterations=10
)


# --- Simple REPL loop ---
if __name__ == "__main__":
    print("🤖 LangChain AI Agent with Tools (Ollama)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        response = agent.invoke(user_input)
        print(f"AI: {response}\n")