import requests
import re
from tools import calculator_tool, search_tool

import os
from dotenv import load_dotenv
LAAMA_3B = os.getenv("LAAMA_3B")
OLLAMA_URL = os.getenv("OLLAMA_URL")

# 🧠 System Prompt
SYSTEM_PROMPT = """You are an AI agent with access to tools.

Available tools:
- [Calculator](expression): For math (e.g. 2 + 2, 9 * 5)
- [Search](query): For current information

Only respond with a tool if it is absolutely needed. Do NOT invent tools like <Hello> or wrap greetings in tags.

If no tool is needed, just reply naturally like a regular assistant."""


def query_ollama(prompt: str, model=MODEL) -> str:
    """
    Sends a prompt to the Ollama model.
    """
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json()["response"].strip()


def build_prompt(system_prompt, chat_history, user_input):
    history_text = "\n".join(chat_history)
    return f"{system_prompt}\n\n{history_text}\nUser: {user_input}\nAI:"


def run_agent(user_input: str, chat_history: list[str], model=MODEL):
    # Step 1: Ask model what to do
    full_prompt = build_prompt(SYSTEM_PROMPT, chat_history, user_input)
    response = query_ollama(full_prompt, model)

    # print(f"\n🧠 AI: {response}")
    tool_call = re.search(r"<tool>\[(.*?)\]\((.*?)\)</tool>", response)

    if tool_call:
        tool_name = tool_call.group(1).strip()
        tool_input = tool_call.group(2).strip()

        # Run the tool immediately without asking
        if tool_name == "Calculator":
            result = calculator_tool(tool_input)
        elif tool_name == "Search":
            result = search_tool(tool_input)
        else:
            result = f"[Unknown tool: {tool_name}]"

        # Log everything
        chat_history.append(f"User: {user_input}")
        chat_history.append(f"AI: {response}")
        chat_history.append(f"Tool[{tool_name}]: {result}")

        # Return final answer directly
        return f"\n🤖 I ran the [{tool_name}] tool with input: '{tool_input}'\n📎 Result: {result}"
    
    else:
        # No tool needed — just return normal response
        chat_history.append(f"User: {user_input}")
        chat_history.append(f"AI: {response}")
        return f"\n💬 AI Response , no tools needed: {response}"

    

if __name__ == "__main__":
    history = []
    print("🤖 AI Agent started. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        output = run_agent(user_input, history)
        print("AI:", output)