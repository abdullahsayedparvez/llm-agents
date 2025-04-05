from langchain.agents import Tool


def get_tools(df, llm):
    def get_column_names(_):
        return ", ".join(df.columns.tolist())

    def get_row_count(_):
        return f"The dataset has {len(df)} rows."

    return [
        Tool.from_function(name="get_column_names", func=get_column_names, description="Returns column names from the dataset"),
        Tool.from_function(name="get_row_count", func=get_row_count, description="Returns the number of rows in the dataset"),
    ]
