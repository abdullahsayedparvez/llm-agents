def calculator_tool(input: str) -> str:
    try:
        result = eval(input)
        return f"✅ Result: {result}"
    except Exception as e:
        return f"❌ Error: {e}"


def search_tool(query: str) -> str:
    # Simulated search result
    return f"📰 Simulated search result for '{query}': Tata launches new EV, stocks rise 5%."

# Chat Tool
def chat_tool(input: str) -> str:
    return f"👋 {input.strip().capitalize()}! How can I help you today?"
