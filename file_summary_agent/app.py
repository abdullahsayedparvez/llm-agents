import streamlit as st
from summary_generator import generate_summary
from agent_tools import get_tools
from langchain.llms import Ollama
from langchain.agents import initialize_agent
import os
from dotenv import load_dotenv
import pandas as pd
load_dotenv()


def read_file(uploaded_file):
    file_name = uploaded_file.name.lower()
    try:
        if file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith('.txt'):
            content = uploaded_file.read().decode("utf-8")
            df = pd.DataFrame({'text': content.splitlines()})
        else:
            st.warning("Unsupported file type. Showing raw text.")
            content = uploaded_file.read().decode("utf-8", errors="ignore")
            df = pd.DataFrame({'text': content.splitlines()})
        return df
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return None
    

LAAMA_3B = os.getenv("LAAMA_3B")
OLLAMA_URL = os.getenv("OLLAMA_URL")
st.set_page_config(page_title="📊 Excel AI Assistant", layout="wide")
st.title("📊 Excel AI Assistant (Summary + Agent Chat)")
uploaded_file = st.file_uploader("Upload any file", type=None)


if uploaded_file:
    df = read_file(uploaded_file)
    st.session_state["excel_df"] = df

    st.subheader("👀 Preview of your data")
    st.dataframe(st.session_state["excel_df"].head())

    # Generate summary
    if "excel_summary" not in st.session_state:
        with st.spinner("Generating summary..."):
            summary = generate_summary(df)
            st.session_state["excel_summary"] = summary

    st.subheader("📋 Summary")
    st.write(st.session_state["excel_summary"])

    # Setup agent
    llm = Ollama(model=LAAMA_3B, base_url=OLLAMA_URL)
    tools = get_tools(df, llm)

    if "agent" not in st.session_state:
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent="zero-shot-react-description",
            verbose=True,
            max_iterations=15,  # or higher like 15
            handle_parsing_errors=True
            
        )
        st.session_state["agent"] = agent
    # Initialize conversation history
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = []

    # Display conversation history
    for idx, (user_msg, bot_msg) in enumerate(st.session_state["conversation_history"]):
        st.markdown(f"**🧑 You:** {user_msg}")
        st.markdown(f"**🤖 Agent:** {bot_msg}")

    st.markdown("---")

    # Form for user input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask a question about your data:", key="user_input")
        submitted = st.form_submit_button("Send")

    if submitted and user_input:
        with st.spinner("Thinking..."):
            response = st.session_state["agent"].run(user_input)
            st.session_state["conversation_history"].append((user_input, response))

        # Show updated dataframe if changed
        if "excel_df" in st.session_state:
            st.markdown("### 🔄 Updated DataFrame")
            st.dataframe(st.session_state["excel_df"].head())