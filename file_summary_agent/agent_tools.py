from langchain.agents import Tool
import streamlit as st


def get_tools(df, llm):
    def get_column_names(_):
        return ", ".join(df.columns.tolist())

    def get_row_count(_):
        return f"The dataset has {len(df)} rows."

    def modify_dataframe(command: str):
        try:
            local_df = df.copy()
            exec(command, {"df": local_df})
            return str(local_df.head(5))
        except Exception as e:
            return f"Error executing command: {e}"

    def interpret_and_execute(user_input: str):
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain

        # Prompt the LLM to convert the user input into a Pandas command
        prompt = PromptTemplate.from_template("""
                        You are an expert data analyst. Convert the user's request into a valid Python Pandas command to be executed on a DataFrame named `df`.

                        Only return valid Python code. Do not include explanations.

                        User request: {user_input}
                        """)

        chain = LLMChain(llm=llm, prompt=prompt)
        pandas_code = chain.run(user_input)
        try:
            # 🧠 Get current df from session
            local_df = st.session_state["excel_df"]

            # 👀 Execute command on the dataframe
            exec(pandas_code, {"df": local_df})

            # 📝 Save updated DataFrame back to session
            st.session_state["excel_df"] = local_df

            return f"✅ Done! Here's the updated data:\n\n{local_df.head().to_markdown()}"
        except Exception as e:
            return f"⚠️ Error running the command:\n```\n{e}\n```"

    return [
        Tool.from_function(
            name="get_column_names",
            func=get_column_names,
            description="Use this tool when the user asks for column names, headers, or fields in the dataset."
        ),
        Tool.from_function(
            name="get_row_count",
            func=get_row_count,
            description="Use this tool when the user wants to know how many rows are in the dataset or how many records there are."
        ),
        Tool.from_function(
            name="dataframe_transform",
            func=interpret_and_execute,
            description="Use this tool to modify or transform the DataFrame based on user requests. This includes renaming columns, changing text case, filtering rows, sorting, filling missing values, etc."
        )
    ]
