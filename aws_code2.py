import pandas as pd
import numpy as np
import faiss
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import re  # For parsing query in hardcoded tool
from crewai import Agent, Task, Crew, Process  # CrewAI imports
from crewai.tools import tool  # Import for decorator
from langchain_aws import ChatBedrock  # For Bedrock LLM integration with CrewAI
from langchain_aws import BedrockEmbeddings  # For Bedrock embeddings
from dotenv import load_dotenv
import os
import boto3

# Load environment variables from .env file
load_dotenv()

# AWS credentials from .env
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region_name = os.getenv("AWS_REGION_NAME")  # e.g., 'us-east-1'

# Create Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region_name,
)

# LLM setup for CrewAI (using Amazon Titan Text Express as proxy for "nova"; adjust model_id if needed)
llm = ChatBedrock(
    model_id="amazon.titan-text-express-v1",  # Assuming "nova" refers to a Titan variant; change to e.g., "anthropic.claude-3-sonnet-20240229-v1:0" if Claude is intended
    client=bedrock_client,
    temperature=0.7,
)

# Embeddings setup using Amazon Titan Text Embeddings
embedder = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", client=bedrock_client
)

# Paths to the files (adjust if needed)
data_dict_path = "data_dict.xlsx"
dataset_path = "synthetic_dataset_500.xlsx"


# Tool: Build FAISS DB (decorated)
@tool("Build FAISS DB")
def build_faiss_db() -> str:
    """Builds the FAISS vector database from the data dictionary Excel file."""
    sheet1 = pd.read_excel(data_dict_path, sheet_name="Sheet1")
    sheet2 = pd.read_excel(data_dict_path, sheet_name="Sheet2")
    documents = []
    metadata = []
    for _, row in sheet1.iterrows():
        key = row.get("NAME", "")
        desc = row.get("DESCRIPTION", "")
        units = row.get("UNITS", "")
        valid_values = row.get("VALID VALUES", "")
        text = f"Key: {key}, Description: {desc}, Units: {units}, Valid Values: {valid_values}"
        documents.append(text)
        metadata.append({"sheet": "Sheet1", "key": key})
    for _, row in sheet2.iterrows():
        key = row.get("Name", "")
        desc = row.get("Description", "")
        code_desc = row.get("Code  Description", "")
        text = f"Key: {key}, Description: {desc}, Code Description: {code_desc}"
        documents.append(text)
        metadata.append({"sheet": "Sheet2", "key": key})
    embeddings = embedder.embed_documents(documents)
    dimension = len(embeddings[0])  # Get dimension from first embedding
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, "faiss_index.index")
    pd.DataFrame({"documents": documents, "metadata": metadata}).to_json(
        "metadata.json", orient="records"
    )
    return "FAISS DB built successfully."


# Tool: Query FAISS DB (decorated)
@tool("Query FAISS DB")
def query_faiss(query: str, top_k: int = 5) -> list:
    """Queries the FAISS DB for similar documents based on the query."""
    index = faiss.read_index("faiss_index.index")
    metadata_df = pd.read_json("metadata.json")
    query_emb = embedder.embed_query(query)
    distances, indices = index.search(np.array([query_emb]).astype("float32"), top_k)
    results = []
    for idx in indices[0]:
        results.append(
            {
                "document": metadata_df.iloc[idx]["documents"],
                "metadata": metadata_df.iloc[idx]["metadata"],
            }
        )
    return results


# Hardcoded Tool: Perform calculations and plot with templated code (decorated)
@tool("Perform Calculation and Plot")
def perform_calculation_and_plot_hardcoded(
    user_query: str, variable_info: list = None
) -> str:
    """Performs calculations (using LLM-inferred stats like mean, sum) and plots based on the query and variable info."""
    # Load dataset
    dataset = pd.read_excel(dataset_path, sheet_name="Sheet1")

    # Use LLM to determine the operation (aggregation function)
    op_prompt = f"From the query '{user_query}', determine the pandas aggregation function to use (e.g., mean, sum, count, median, max, min, std, var). Return only the function name."
    try:
        operation = llm.invoke(op_prompt).content.strip().lower()
    except Exception as e:
        operation = "mean"  # Fallback on LLM error

    # Fallback if LLM fails or invalid
    if operation not in ["mean", "sum", "count", "median", "max", "min", "std", "var"]:
        operation = "mean"  # Default

    # Parse query for groupby
    groupby_col = re.search(r"(?:by|across)\s+(\w+)", user_query.lower())
    groupby_col = (
        groupby_col.group(1) if groupby_col else "product"
    )  # Default to 'product'

    # Extract key variables from query or use variable_info from DB
    key_vars = re.findall(r"(P13_\w+)", user_query.upper())
    if not key_vars and variable_info:
        # Extract from DB results
        for res in variable_info:
            key_match = re.search(r"Key:\s*(P13_\w+)", res["document"])
            if key_match:
                key_vars.append(key_match.group(1))
    if not key_vars:
        return "No valid key variables found in query or DB."

    # Clean product column if grouping by 'product'
    if groupby_col == "product":
        dataset["clean_product"] = dataset["product"].str.extract(
            r"([A-Z\s]+)$"
        )  # Extract e.g., 'RETAIL CARD'
        groupby_col = "clean_product"

    # Hardcoded code template (flexible for any operation)
    code_template = """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assume dataset is loaded
grouped = dataset.groupby('{groupby_col}')['{key_vars}']
plot_data = grouped.agg('{operation}').reset_index()
plot_data['Result'] = plot_data.mean(axis=1, numeric_only=True) if len(plot_data.columns) > 2 else plot_data['{key_vars}'.split(', ')[0]]

# Plot (mimic uploaded style: blue bars, vertical, with values on top)
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='{groupby_col}', y='Result', data=plot_data, color='blue')
plt.title('{op_cap} {title_desc} by {groupby_col.capitalize()}')
plt.xlabel('{groupby_col.capitalize()} Category')
plt.ylabel('{op_cap} {units}')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, plot_data['Result'].max() * 1.2)

# Add value labels on bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')

plt.tight_layout()
"""
    # Determine title/units from DB or query
    title_desc = "Value"
    units = ""
    if variable_info:
        for res in variable_info:
            desc_match = re.search(r"Description:\s*(.*?),", res["document"])
            units_match = re.search(r"Units:\s*([^,]+)", res["document"])
            if desc_match:
                title_desc = desc_match.group(1)
            if units_match:
                units = units_match.group(1)
    else:
        title_desc = (
            user_query.split(f"{operation.capitalize()} ")[-1].split(" across")[0]
            if " " in user_query
            else "Value"
        )
        units = "$"  # Default

    # Fill template
    filled_code = code_template.format(
        groupby_col=groupby_col,
        key_vars="', '".join(key_vars),
        operation=operation,
        op_cap=operation.capitalize(),
        title_desc=title_desc,
        units=units,
    )

    # Execute
    try:
        local_vars = {"dataset": dataset, "plt": plt, "sns": sns, "pd": pd}
        exec(filled_code, globals(), local_vars)

        # Save and show plot
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = Image.open(buf)
        img.show()  # Pops up the image; or save: plt.savefig('plot.png')

        # Generate table for summary
        plot_data = local_vars["plot_data"]  # Access from exec locals
        summary_table = plot_data[[groupby_col, "Result"]].to_markdown(index=False)
        summary = (
            f"Summary Table:\n{summary_table}\n\nThe bar chart has been displayed."
        )
        return summary
    except Exception as e:
        return f"Error executing template: {e}"


# CrewAI Setup: Agents
db_agent = Agent(
    role="Database Query Agent",
    goal="Build and query the FAISS vector DB to find relevant columns based on descriptions.",
    backstory="Expert in semantic search for data dictionaries to identify metrics like column names from descriptions.",
    tools=[build_faiss_db, query_faiss],
    llm=llm,
    verbose=True,
)

analysis_agent = Agent(
    role="Data Analysis Agent",
    goal="Calculate statistics (e.g., mean, sum, count) grouped by categories and generate plots using hardcoded templates. Use LLM to infer operations.",
    backstory="Data analyst using LLM to determine aggregations and templated code for reliable computations and visualizations.",
    tools=[perform_calculation_and_plot_hardcoded],
    llm=llm,
    verbose=True,
)

# Tasks
task_build_db = Task(
    description="Build the FAISS DB if not already built.",
    expected_output="Confirmation that DB is built.",
    agent=db_agent,
)

task_query_db = Task(
    description='Query the FAISS DB using the user query description to find the matching column (e.g., for "Total credit amount on open Auto lease trades reported in the last 3 months", find P13_AUL5320). User query: {user_query}.',
    expected_output="List of relevant variable keys and descriptions.",
    agent=db_agent,
    output_file="variable_info.json",  # Save for next task
)

task_analyze_and_plot = Task(
    description="Use the hardcoded tool to calculate the inferred statistic (e.g., mean, sum) grouped by product (or specified) and plot a bar chart like the sample. Pass variable_info from previous task to select columns if needed. User query: {user_query}.",
    expected_output="Summary of calculations, table, and confirmation of plot generation.",
    agent=analysis_agent,
    context=[task_query_db],  # Use output from DB query
)

# Crew
crew = Crew(
    agents=[db_agent, analysis_agent],
    tasks=[task_build_db, task_query_db, task_analyze_and_plot],
    process=Process.sequential,
    verbose=True,  # Changed from 2 to True
)

# Main
if __name__ == "__main__":
    user_query = "Average Total credit amount on open Auto lease trades reported in the last 3 months across all products"

    result = crew.kickoff(inputs={"user_query": user_query})
    print("Crew Result:", result)
