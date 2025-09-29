import os
import pickle
import re
import sys
from io import StringIO
import json
from dotenv import load_dotenv
import faiss
import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# AWS Bedrock configuration from environment variables
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# ==============================================================================
# PART 1: KNOWLEDGE AGENT (Updated for AWS Bedrock)
# ==============================================================================

class KnowledgeAgent:
    def __init__(self, excel_file, force_recreate=False):
        self.excel_file = excel_file
        self.faiss_index_file = "my_vector_db.index"
        self.knowledge_base_file = "knowledge_base.pkl"

        # Initialize AWS Bedrock client for embeddings
        try:
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=AWS_REGION,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            )
        except Exception as e:
            print(f"❌ Error initializing AWS Bedrock client: {e}")
            sys.exit(1)

        if force_recreate or not os.path.exists(self.faiss_index_file):
            self._create_and_save_knowledge_base()
        else:
            self._load_knowledge_base()

    def _get_embeddings(self, texts):
        """Generate embeddings using Amazon Titan Text Embeddings V2"""
        embeddings_list = []

        for text in texts:
            try:
                body = json.dumps({
                    "inputText": text,
                    "dimensions": 512,  # Using 512 dimensions for efficiency
                    "normalize": True
                })

                response = self.bedrock_client.invoke_model(
                    modelId='amazon.titan-embed-text-v2:0',
                    contentType='application/json',
                    accept='application/json',
                    body=body
                )

                response_body = json.loads(response['body'].read())
                embedding = np.array(response_body['embedding'], dtype=np.float32)
                embeddings_list.append(embedding)

            except Exception as e:
                print(f"❌ Error generating embedding for text: {text[:50]}... Error: {e}")
                # Create a zero embedding as fallback
                embeddings_list.append(np.zeros(512, dtype=np.float32))

        return np.array(embeddings_list)

    def _create_and_save_knowledge_base(self):
        print("--- 🧠 Building knowledge base from Excel file... ---")

        try:
            df_attributes = pd.read_excel(self.excel_file, sheet_name="Sheet1")
            df_prefixes = pd.read_excel(self.excel_file, sheet_name="Sheet2")
        except Exception as e:
            print(f"❌ Error reading '{self.excel_file}'. Please check sheet and file names. Details: {e}")
            sys.exit(1)

        prefix_to_name_map = pd.Series(
            df_prefixes["Industry name"].values, index=df_prefixes["Industry prefix"]
        ).to_dict()

        self.knowledge_base = []
        full_text_list = []
        self.category_map = {name.lower(): name for name in prefix_to_name_map.values()}

        for index, row in df_attributes.iterrows():
            variable_name = row["NAME"]
            description = row["DESCRIPTION"]
            defaults = row.get("DEFAULT VALUES & DISPLAY", "Not specified")

            category_name = "Unknown"
            match = re.search(r"P\d{2}_([A-Z]{3})", variable_name)
            if match:
                prefix = match.group(1)
                category_name = prefix_to_name_map.get(prefix, "Unknown")

            entry = {
                "Variable Name": variable_name,
                "Description": description,
                "Defaults": defaults,
                "Category": category_name,
            }
            self.knowledge_base.append(entry)

            full_text = f"{category_name}. {description}. Variable Name is {variable_name}."
            full_text_list.append(full_text)

        print(" - Generating embeddings using Amazon Titan Text Embeddings V2...")
        embeddings = self._get_embeddings(full_text_list)

        print(" - Creating and saving FAISS index...")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, self.faiss_index_file)
        self.index = index

        with open(self.knowledge_base_file, "wb") as f:
            pickle.dump((self.knowledge_base, self.category_map), f)

        print("--- ✅ Knowledge base created successfully! ---")

    def _load_knowledge_base(self):
        print("--- 🧠 Loading existing knowledge base from files... ---")
        self.index = faiss.read_index(self.faiss_index_file)
        with open(self.knowledge_base_file, "rb") as f:
            self.knowledge_base, self.category_map = pickle.load(f)
        print("--- ✅ Knowledge base loaded successfully! ---")

    def find_relevant_variable(self, query: str):
        print(f" 🔎 [Tool Call] Searching knowledge base for: '{query}'")

        query_lower = query.lower()
        query_category = None

        # Check for category match
        for cat_keyword in self.category_map.keys():
            if re.search(r"\b" + re.escape(cat_keyword) + r"\b", query_lower):
                query_category = self.category_map[cat_keyword]
                break

        # Keyword-based matching
        best_keyword_match = None
        highest_score = 0
        query_words = set(query_lower.split())

        for item in self.knowledge_base:
            description_words = set(item["Description"].lower().split())
            intersection = len(query_words.intersection(description_words))
            union = len(query_words.union(description_words))
            score = intersection / union if union != 0 else 0

            if query_category and item["Category"].lower() != query_category.lower():
                continue

            if score > highest_score:
                highest_score = score
                best_keyword_match = item

        if highest_score > 0.4:
            return best_keyword_match

        # Vector similarity search
        try:
            query_embeddings = self._get_embeddings([query])
            _, indices = self.index.search(query_embeddings, 1)
            if indices.size > 0:
                return self.knowledge_base[indices[0][0]]
        except Exception as e:
            print(f"❌ Error in vector search: {e}")

        return None

# ==============================================================================
# PART 2: THE MULTI-STEP REASONING AGENT (Updated for AWS Bedrock)
# ==============================================================================

class DataAnalysisAgent:
    def __init__(self, knowledge_tool: KnowledgeAgent):
        self.knowledge_tool = knowledge_tool

        try:
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=AWS_REGION,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            )
        except Exception as e:
            print(f"❌ Critical Error: Could not configure AWS Bedrock client. Details: {e}")
            sys.exit(1)

    def _generate_response(self, prompt):
        """Generate response using Amazon Nova Micro model"""
        try:
            body = json.dumps({
                "schemaVersion": "messages-v1",
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {
                    "max_new_tokens": 2000,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            })

            response = self.bedrock_client.invoke_model(
                modelId='amazon.nova-micro-v1:0',
                contentType='application/json',
                accept='application/json',
                body=body
            )

            response_body = json.loads(response['body'].read())
            return response_body['output']['message']['content'][0]['text']

        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return "Error generating response"

    def _execute_analysis_code(self, code, data_path, chart_path):
        summary = None
        local_namespace = {"plt": plt, "pd": pd, "data_path": data_path}

        try:
            final_code = code.replace("CHART_PATH_PLACEHOLDER", chart_path)
            exec(final_code, globals(), local_namespace)

            if "final_answer" in local_namespace:
                summary = str(local_namespace["final_answer"])
        except SyntaxError as e:
            summary = f"Syntax Error: The AI generated invalid Python code. Error: {e}"
        except Exception as e:
            summary = f"An error occurred during code execution: {e}"

        return summary

    def run(self, user_query, data_path):
        print(f"\n🤔 **Agent Initializing for query:** '{user_query}'")
        chart_path = os.path.abspath("chart.png").replace("\\", "/")

        # --- Step 1: Create a Plan ---
        print("\n--- Step 1: Creating a plan ---")
        planning_prompt = f"""
You are a data analysis planner. Your job is to break down a user's query into complete, self-contained search queries for a tool called `find_variable`.

User Query: "{user_query}"

Based on the query, create a list of the full descriptions of the variables you need to find. Each item should be on a new line and should be a complete thought.
"""

        plan_response = self._generate_response(planning_prompt)
        search_queries = [line.strip() for line in plan_response.strip().split("\n") if line.strip()]

        print("🤖 Agent's Plan:")
        for query in search_queries:
            print(f" - Find variable for: '{query}'")

        # --- Step 2: Execute the Plan (Use Tools) ---
        print("\n--- Step 2: Executing the plan (using tools) ---")
        tool_results = []

        for query in search_queries:
            result = self.knowledge_tool.find_relevant_variable(query)
            tool_results.append(f"Search Query: '{query}'\nResult: {result}\n")

        print("✅ Tool execution complete.")

        # --- Step 3: Synthesize the Final Code ---
        print("\n--- Step 3: Generating the final analysis code ---")
        synthesis_prompt = f"""
You are a data analysis code generator. Your job is to write a final Python script to answer the user's query based on the results from your tool calls.

User Query: "{user_query}"

Data Path: "{data_path}"

Chart Save Path: "CHART_PATH_PLACEHOLDER"

Tool Results:
{''.join(tool_results)}

Based on the User Query and the variables found in the Tool Results, write a complete Python script to perform the analysis.

- The data is a CSV file. You MUST load it using `pd.read_csv(data_path, sep=',', encoding='latin1')`.
- Clean the data by removing default/special values before analysis.
- Assign your final, descriptive string answer to a variable named `final_answer`.
- If a chart is required, you MUST save it using `plt.savefig('CHART_PATH_PLACEHOLDER')`.
- Your output must be ONLY the raw Python code. Do not use markdown.
"""

        final_code = self._generate_response(synthesis_prompt).strip().replace("```python", "").replace("```", "")

        print("\n💻 **Final Generated Code:**")
        print("-" * 30)
        print(final_code)
        print("-" * 30)

        # --- Step 4: Execute the Final Code ---
        print("\n Executing final code...")
        summary = self._execute_analysis_code(final_code, data_path, chart_path)

        if summary:
            print("\n✅ **Agent's Final Answer:**")
            print(summary)
            if os.path.exists(chart_path):
                print(f" A chart visualizing the data has been saved to '{chart_path}'.")
        else:
            print("\n❌ **Agent's Conclusion:** The analysis ran, but no final answer was produced.")

        return

# ==============================================================================
# --- 🚀 MAIN EXECUTION ---
# ==============================================================================

if __name__ == "__main__":
    # Check if environment variables are set
    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
        print("\n\n!!! 🛑 IMPORTANT 🛑 !!!")
        print("Please set your AWS credentials in the .env file:")
        print("AWS_ACCESS_KEY_ID=your_access_key")
        print("AWS_SECRET_ACCESS_KEY=your_secret_key")
        print("AWS_REGION=your_region (optional, defaults to us-east-1)")
    else:
        dictionary_excel_file = "data_dict.xlsx"
        analysis_csv_file = "synthetic_dataset_500.xlsx"

        knowledge_agent = KnowledgeAgent(
            excel_file=dictionary_excel_file, force_recreate=False
        )

        analysis_agent = DataAnalysisAgent(knowledge_tool=knowledge_agent)

        print("\n--- 🤖 Multi-Step Reasoning Agent is Ready ---")
        print("--- Type 'exit' or 'quit' to end the session. ---")

        while True:
            user_question = input(f"\nYour question about '{analysis_csv_file}': ")

            if user_question.lower() in ["exit", "quit"]:
                print("\nGoodbye! 👋")
                break

            # Clear the old chart before running new analysis
            if os.path.exists("chart.png"):
                os.remove("chart.png")

            analysis_agent.run(user_question, data_path=analysis_csv_file)
