import pandas as pd
import numpy as np
import faiss
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import re
import os
import boto3
import json
from typing import List, Dict, Any

# CrewAI imports with proper LLM setup
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from crewai_tools import SerperDevTool

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

print("=" * 70)
print("AWS BEDROCK CREWAI SYSTEM - TITAN EMBEDDING & NOVA MICRO")
print("=" * 70)

# AWS Bedrock Configuration
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region_name = os.getenv("AWS_REGION_NAME", "us-east-1")

if not aws_access_key_id or not aws_secret_access_key:
    raise ValueError("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your .env file")

# Configure AWS Bedrock LLM for CrewAI
print("Configuring AWS Bedrock Nova Micro LLM for CrewAI...")
llm = LLM(
    model="bedrock/amazon.nova-micro-v1:0",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_region_name=aws_region_name,
    temperature=0.1,
)

print("✅ AWS Bedrock Nova Micro LLM configured successfully")

# Initialize Bedrock clients for embeddings
print("Setting up AWS Bedrock clients...")
bedrock_runtime = boto3.client(
    'bedrock-runtime',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region_name
)
print("✅ AWS Bedrock runtime client initialized")

# Embedding function using Amazon Titan Embed Text v2
def get_bedrock_embeddings(texts):
    """Generate embeddings using Amazon Titan Embed Text v2"""
    embeddings = []

    for text in texts:
        try:
            # Prepare the request for Titan Embed Text v2
            body = json.dumps({
                "inputText": text,
                "dimensions": 1024,  # Use default 1024 dimensions
                "normalize": True
            })

            # Call Bedrock to get embeddings
            response = bedrock_runtime.invoke_model(
                modelId='amazon.titan-embed-text-v2:0',
                body=body,
                contentType='application/json',
                accept='application/json'
            )

            # Parse the response
            response_body = json.loads(response['body'].read())
            embedding = response_body['embedding']
            embeddings.append(embedding)

        except Exception as e:
            print(f"Error getting embedding for text: {str(e)}")
            # Fallback to zero vector if embedding fails
            embeddings.append([0.0] * 1024)

    return np.array(embeddings)

print(f"✅ Amazon Titan Embed Text v2 embedding function ready (1024D)")

# File paths
data_dict_path = "data_dict.xlsx"
dataset_path = "synthetic_dataset_500.xlsx"
faiss_index_path = "enhanced_faiss_index.index"
metadata_path = "enhanced_metadata.json"

# ===============================
# TOOLS - Enhanced with FAISS persistence check and Bedrock embeddings
# ===============================

@tool("Build FAISS Database")
def build_faiss_db() -> str:
    """Builds the FAISS vector database from the data dictionary Excel file with enhanced metadata.
    Uses Amazon Titan Embed Text v2 for embeddings. Checks if database already exists and skips rebuild if present."""

    # Check if FAISS database already exists
    if os.path.exists(faiss_index_path) and os.path.exists(metadata_path):
        try:
            # Try to load existing database to verify it's valid
            index = faiss.read_index(faiss_index_path)
            metadata_df = pd.read_json(metadata_path)
            print(f"✅ FAISS database already exists with {index.ntotal} documents")
            print(f"✅ Using existing database: {faiss_index_path}")
            return f"FAISS database already exists with {index.ntotal} documents. Using existing database."
        except Exception as e:
            print(f"⚠️ Existing database corrupted: {str(e)}")
            print("Building new database...")
    else:
        print("FAISS database not found. Building new database...")

    try:
        print("Building FAISS database with Amazon Titan embeddings...")
        documents = []
        metadata = []

        # Process Sheet1
        try:
            sheet1 = pd.read_excel(data_dict_path, sheet_name="Sheet1")
            print(f"Processing Sheet1: {len(sheet1)} rows")
        except FileNotFoundError:
            return f"Error: File {data_dict_path} not found. Please ensure the Excel file exists."
        except Exception as e:
            return f"Error reading Sheet1: {str(e)}"

        for _, row in sheet1.iterrows():
            key = str(row.get("NAME", "")).strip()
            desc = str(row.get("DESCRIPTION", "")).strip()
            units = str(row.get("UNITS", "")).strip()
            valid_values = str(row.get("VALID VALUES", "")).strip()

            # Enhanced text formatting for better semantic search
            text_parts = []
            if key and key != "nan":
                text_parts.append(f"Variable: {key}")
            if desc and desc != "nan":
                text_parts.append(f"Description: {desc}")
            if units and units != "nan":
                text_parts.append(f"Units: {units}")
            if valid_values and valid_values != "nan":
                text_parts.append(f"Valid values: {valid_values}")

            if text_parts:
                text = ". ".join(text_parts)
                documents.append(text)
                metadata.append({
                    "sheet": "Sheet1",
                    "key": key,
                    "description": desc,
                    "units": units,
                    "valid_values": valid_values,
                })

        # Process Sheet2
        try:
            sheet2 = pd.read_excel(data_dict_path, sheet_name="Sheet2")
            print(f"Processing Sheet2: {len(sheet2)} rows")
        except Exception as e:
            print(f"Warning: Could not read Sheet2: {str(e)}")
            sheet2 = pd.DataFrame()  # Empty dataframe if Sheet2 doesn't exist

        for _, row in sheet2.iterrows():
            key = str(row.get("Name", "")).strip()
            desc = str(row.get("Description", "")).strip()
            code_desc = str(row.get("Code Description", "")).strip()

            text_parts = []
            if key and key != "nan":
                text_parts.append(f"Variable: {key}")
            if desc and desc != "nan":
                text_parts.append(f"Description: {desc}")
            if code_desc and code_desc != "nan":
                text_parts.append(f"Code description: {code_desc}")

            if text_parts:
                text = ". ".join(text_parts)
                documents.append(text)
                metadata.append({
                    "sheet": "Sheet2",
                    "key": key,
                    "description": desc,
                    "code_description": code_desc,
                })

        if not documents:
            return "Error: No documents found to index. Please check your Excel file structure."

        print(f"Prepared {len(documents)} documents for embedding...")

        # Generate embeddings using Amazon Titan Embed Text v2
        print("Generating embeddings with Amazon Titan Embed Text v2...")
        embeddings = get_bedrock_embeddings(documents)

        # Create FAISS index (using Inner Product for normalized vectors = cosine similarity)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(np.array(embeddings).astype("float32"))

        # Save index and metadata
        faiss.write_index(index, faiss_index_path)
        metadata_df = pd.DataFrame({
            "documents": documents,
            "metadata": metadata,
            "embedding_model": "amazon.titan-embed-text-v2:0",
        })
        metadata_df.to_json(metadata_path, orient="records", indent=2)

        print(f"✅ FAISS database built with {len(documents)} documents using Amazon Titan Embed Text v2")
        return f"Enhanced FAISS DB built successfully with {len(documents)} documents using Amazon Titan Embed Text v2 embeddings (1024D)."

    except Exception as e:
        error_msg = f"Error building FAISS DB: {str(e)}"
        print(error_msg)
        return error_msg

@tool("Query FAISS Database")
def query_faiss(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Queries the FAISS DB for similar documents using Amazon Titan embeddings."""
    try:
        print(f"Querying FAISS database for: '{query}'")

        # Check if files exist
        try:
            index = faiss.read_index(faiss_index_path)
            metadata_df = pd.read_json(metadata_path)
        except FileNotFoundError:
            return [
                {
                    "error": "FAISS database not found. Please build the database first using build_faiss_db tool."
                }
            ]

        # Generate query embedding using Amazon Titan
        query_emb = get_bedrock_embeddings([query])

        # Search with cosine similarity
        similarities, indices = index.search(
            np.array(query_emb).astype("float32"), top_k
        )

        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if similarity > 0.2:  # Similarity threshold
                result = {
                    "document": metadata_df.iloc[idx]["documents"],
                    "metadata": metadata_df.iloc[idx]["metadata"],
                    "similarity_score": float(similarity),
                    "rank": i + 1,
                }
                results.append(result)

        print(f"✅ Found {len(results)} relevant matches")
        return results

    except Exception as e:
        error_msg = f"Error querying FAISS DB: {str(e)}"
        print(error_msg)
        return [{"error": error_msg}]

@tool("Perform Analysis and Visualization")
def perform_calculation_and_plot(user_query: str, variable_info: str = "") -> str:
    """Enhanced analysis and plotting with proper error handling."""
    try:
        print("Performing enhanced data analysis...")

        # Load dataset
        try:
            dataset = pd.read_excel(dataset_path, sheet_name="Sheet1")
            print(f"Dataset loaded: {len(dataset)} rows")
        except FileNotFoundError:
            return f"Error: Dataset file {dataset_path} not found. Please ensure the Excel file exists."
        except Exception as e:
            return f"Error loading dataset: {str(e)}"

        # Simple operation determination (fallback approach)
        operation_keywords = {
            "average": "mean",
            "mean": "mean",
            "avg": "mean",
            "sum": "sum",
            "total": "sum",
            "count": "count",
            "number": "count",
            "maximum": "max",
            "max": "max",
            "minimum": "min",
            "min": "min",
        }

        operation = "mean"  # default
        for keyword, op in operation_keywords.items():
            if keyword.lower() in user_query.lower():
                operation = op
                break

        print(f"Using aggregation: {operation}")

        # Parse query for groupby
        groupby_match = re.search(r"(?:by|across)\s+(\w+)", user_query.lower())
        groupby_col = groupby_match.group(1) if groupby_match else "product"

        # Extract key variables from query
        key_vars = re.findall(r"(P13_\w+)", user_query.upper())

        # If no variables found in query, try to parse from variable_info
        if not key_vars and variable_info:
            try:
                if isinstance(variable_info, str):
                    # Simple regex to extract variable names
                    key_vars = re.findall(r"P13_\w+", variable_info)
            except:
                pass

        # Fallback: use common credit-related variables if available
        if not key_vars:
            potential_vars = [
                col
                for col in dataset.columns
                if "P13_" in col
                and any(term in col.lower() for term in ["credit", "amount", "balance"])
            ]
            if potential_vars:
                key_vars = potential_vars[:3]  # Use first 3 matches
            else:
                # Use any P13 variables as last resort
                key_vars = [col for col in dataset.columns if col.startswith("P13_")][:3]

        if not key_vars:
            return "No valid key variables found in query, dataset, or variable info."

        # Ensure variables exist in dataset
        available_vars = [var for var in key_vars if var in dataset.columns]
        if not available_vars:
            return f"None of the identified variables {key_vars} exist in the dataset. Available columns: {list(dataset.columns)[:10]}..."

        key_vars = available_vars
        print(f"Using variables: {key_vars}")

        # Clean groupby column if needed
        if groupby_col == "product" and "product" in dataset.columns:
            dataset["clean_product"] = dataset["product"].str.extract(r"([A-Z\s]+)$")
            groupby_col = "clean_product"
        elif groupby_col not in dataset.columns:
            # Find a suitable grouping column
            if "product" in dataset.columns:
                groupby_col = "product"
            else:
                # Use first categorical column
                cat_cols = dataset.select_dtypes(include=["object", "category"]).columns
                if len(cat_cols) > 0:
                    groupby_col = cat_cols[0]
                else:
                    return f"No suitable grouping column found. Available columns: {list(dataset.columns)[:10]}..."

        # Perform calculation
        try:
            # Handle missing values
            dataset_clean = dataset.dropna(subset=key_vars + [groupby_col])
            grouped = dataset_clean.groupby(groupby_col)[key_vars]
            plot_data = grouped.agg(operation).reset_index()

            if len(key_vars) > 1:
                plot_data["Result"] = plot_data[key_vars].mean(axis=1, numeric_only=True)
            else:
                plot_data["Result"] = plot_data[key_vars[0]]

        except Exception as calc_error:
            return f"Calculation error: {str(calc_error)}. Dataset shape: {dataset.shape}, Variables: {key_vars}, Groupby: {groupby_col}"

        if len(plot_data) == 0:
            return "No data available after grouping and aggregation."

        # Create enhanced visualization
        plt.figure(figsize=(14, 8))

        # Professional color scheme
        colors = plt.cm.viridis(np.linspace(0, 1, len(plot_data)))

        try:
            ax = sns.barplot(data=plot_data, x=groupby_col, y="Result", palette=colors)

            # Enhanced styling
            plt.title(
                f'{operation.capitalize()} Analysis by {groupby_col.replace("_", " ").title()}\n'
                f"CrewAI + AWS Bedrock (Nova Micro + Titan Embed v2) Analysis",
                fontsize=16,
                fontweight="bold",
                pad=20,
            )

            plt.xlabel(f'{groupby_col.replace("_", " ").title()} Category', fontsize=12)
            plt.ylabel(f"{operation.capitalize()} Value", fontsize=12)
            plt.xticks(rotation=45, ha="right")

            # Add value labels on bars
            for patch in ax.patches:
                height = patch.get_height()
                if not np.isnan(height) and height != 0:
                    ax.annotate(
                        f"{height:.2f}",
                        (patch.get_x() + patch.get_width() / 2.0, height),
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

            # Add professional styling
            plt.grid(axis="y", alpha=0.3, linestyle="--")
            plt.tight_layout()

            # Save plot
            plt.savefig("crewai_analysis_result.png", dpi=300, bbox_inches="tight")
            print("✅ Visualization saved as 'crewai_analysis_result.png'")
            plt.show()

        except Exception as viz_error:
            return f"Visualization error: {str(viz_error)}. Data shape: {plot_data.shape}"

        # Generate comprehensive summary
        try:
            stats = plot_data["Result"].describe()
            summary_table = plot_data[[groupby_col, "Result"]].to_string(index=False)

            summary = f"""
CrewAI AWS Bedrock Analysis Complete:
====================================
Query: {user_query}
Operation: {operation.capitalize()}
Variables: {', '.join(key_vars)}
Grouping: {groupby_col.replace('_', ' ').title()}
Data Points: {len(plot_data)} categories

Results Summary:
{summary_table}

Statistical Overview:
- Count: {stats['count']:.0f} categories
- Mean: {stats['mean']:.2f}
- Standard Deviation: {stats['std']:.2f}
- Minimum: {stats['min']:.2f}
- Maximum: {stats['max']:.2f}

AWS Bedrock Technology Stack:
- AI Framework: CrewAI with LiteLLM integration
- LLM: Amazon Nova Micro v1 (128K context)
- Embedding Model: Amazon Titan Embed Text v2 (1024D)
- Vector Database: FAISS with cosine similarity
- Visualization: Professional matplotlib/seaborn charts
- AWS Region: {aws_region_name}

Output: 'crewai_analysis_result.png' saved successfully
"""
            return summary

        except Exception as summary_error:
            return f"Summary generation error: {str(summary_error)}. Analysis completed but summary failed."

    except Exception as e:
        error_msg = f"Analysis error: {str(e)}"
        print(error_msg)
        return error_msg

# ===============================
# CREWAI AGENTS - AWS Bedrock configuration
# ===============================

class AWSBedrockDataAnalysisCrew:
    """AWS Bedrock CrewAI system with FAISS persistence and continuous queries"""

    def __init__(self):
        print("Initializing AWS Bedrock CrewAI Data Analysis System...")

    def database_researcher(self) -> Agent:
        """Database research agent with AWS Bedrock LLM configuration"""
        return Agent(
            role="Senior Database Research Specialist",
            goal="Build and query FAISS vector databases using Amazon Titan embeddings to find relevant data variables based on semantic similarity",
            backstory="""You are a senior data engineer with over 10 years of experience in semantic search
            and vector databases. You excel at building FAISS indexes with Amazon Bedrock embeddings and finding 
            the most relevant variables from data dictionaries using advanced AWS embedding techniques.""",
            llm=llm,  # Using the AWS Bedrock Nova Micro LLM
            verbose=True,
            allow_delegation=False,
            max_iter=20,
            tools=[build_faiss_db, query_faiss],
            max_retry_limit=2,
        )

    def data_analysis_specialist(self) -> Agent:
        """Data analysis agent with AWS Bedrock LLM configuration"""
        return Agent(
            role="Senior Data Analysis Specialist",
            goal="Perform statistical analysis and create compelling visualizations using AWS Bedrock intelligence",
            backstory="""You are an expert data analyst with deep knowledge of statistical methods and
            data visualization powered by AWS Bedrock. You create professional-grade charts and comprehensive 
            analytical reports using Amazon Nova's advanced reasoning capabilities.""",
            llm=llm,  # Using the AWS Bedrock Nova Micro LLM
            verbose=True,
            allow_delegation=False,
            max_iter=15,
            tools=[perform_calculation_and_plot],
            max_retry_limit=2,
        )

    def research_variables_task(self) -> Task:
        """Database research task using AWS Bedrock"""
        return Task(
            description="""Build the FAISS vector database with Amazon Titan embeddings if not already built, 
            then query it to find the most relevant variables for the user query: {user_query}

            Use Amazon Titan Embed Text v2 for semantic similarity matching. Focus on finding variables 
            that match the semantic meaning of the query, especially those related to credit amounts, 
            auto lease trades, and time-based filters.

            Provide the results as a clear list of variables with their descriptions and relevance scores.""",
            expected_output="""A detailed list of relevant variables with their metadata including:
            - Variable names (e.g., P13_AUL5320)
            - Descriptions and units
            - Similarity scores from Amazon Titan embeddings
            - Sheet sources

            Format as a structured summary that can be used by the analysis specialist.""",
            agent=self.database_researcher(),
        )

    def analysis_task(self) -> Task:
        """Data analysis and visualization task using AWS Bedrock"""
        return Task(
            description="""Using the variables identified from the database research powered by Amazon Titan embeddings, 
            perform comprehensive statistical analysis for the user query: {user_query}

            Leverage Amazon Nova Micro's reasoning capabilities to:
            1. Use the variable information from the research task
            2. Determine the appropriate aggregation function
            3. Group the data appropriately
            4. Calculate the statistics
            5. Create a professional visualization
            6. Generate a comprehensive summary report""",
            expected_output="""A complete analysis report including:
            - Statistical summary table with key metrics
            - Professional visualization (saved as PNG)
            - Key insights and findings from the data powered by AWS Bedrock
            - Technical details about the AWS Bedrock analysis approach
            - Business recommendations based on Amazon Nova's reasoning""",
            agent=self.data_analysis_specialist(),
            context=[self.research_variables_task()],  # Use output from research task
        )

    def create_crew(self) -> Crew:
        """Create the AWS Bedrock crew with proper configuration"""
        return Crew(
            agents=[self.database_researcher(), self.data_analysis_specialist()],
            tasks=[self.research_variables_task(), self.analysis_task()],
            process=Process.sequential,
            verbose=True,
            max_rpm=10,  # Conservative rate limiting for Bedrock
            # Configure AWS Bedrock embeddings for CrewAI memory
            embedder={
                "provider": "aws_bedrock",
                "config": {
                    "model": "amazon.titan-embed-text-v2:0",
                    "aws_access_key_id": aws_access_key_id,
                    "aws_secret_access_key": aws_secret_access_key,
                    "aws_region_name": aws_region_name,
                }
            }
        )

# ===============================
# CONTINUOUS QUERY PROCESSING
# ===============================

def process_single_query(crew: Crew, user_query: str):
    """Process a single user query with the AWS Bedrock CrewAI system"""
    try:
        print(f"\n{'='*50}")
        print(f"Processing Query: {user_query}")
        print(f"{'='*50}")

        # Execute the crew
        result = crew.kickoff(inputs={"user_query": user_query})

        print("\n" + "=" * 70)
        print("AWS BEDROCK CREWAI ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nFinal Result:")
        print(result)

        return result

    except Exception as e:
        error_msg = f"Error executing crew for query '{user_query}': {str(e)}"
        print(f"\n❌ {error_msg}")
        return None

def continuous_query_loop():
    """Main function that handles continuous user queries with AWS Bedrock"""
    print("\n" + "=" * 70)
    print("AWS BEDROCK CREWAI DATA ANALYSIS SYSTEM")
    print("Features: FAISS Persistence + Continuous Queries + Amazon Bedrock")
    print("LLM: Amazon Nova Micro v1 | Embeddings: Amazon Titan Embed Text v2")
    print("=" * 70)

    try:
        # Initialize the crew system once
        crew_system = AWSBedrockDataAnalysisCrew()
        crew = crew_system.create_crew()
        print("\n✅ AWS Bedrock CrewAI system initialized successfully")

        print("\n" + "=" * 50)
        print("CONTINUOUS QUERY MODE ACTIVATED")
        print("Enter your queries below (type 'quit', 'exit', or 'stop' to end)")
        print("=" * 50)

        query_count = 0

        while True:
            try:
                # Get user input
                user_query = input("\n🤖 Enter your query: ").strip()

                # Check for exit commands
                if user_query.lower() in ['quit', 'exit', 'stop', 'q']:
                    print("\n👋 Thank you for using the AWS Bedrock CrewAI Data Analysis System!")
                    break

                # Skip empty queries
                if not user_query:
                    print("⚠️  Please enter a valid query.")
                    continue

                query_count += 1
                print(f"\n📊 Processing Query #{query_count} with AWS Bedrock...")

                # Process the query
                result = process_single_query(crew, user_query)

                if result:
                    print("\n📁 Generated Files:")
                    print(" • crewai_analysis_result.png - Professional visualization")
                    if query_count == 1:  # Only show these for first query
                        print(" • enhanced_faiss_index.index - FAISS vector database (Titan embeddings)")
                        print(" • enhanced_metadata.json - Database metadata")

                    print(f"\n✅ Query #{query_count} completed successfully with AWS Bedrock!")
                else:
                    print(f"\n❌ Query #{query_count} failed. Please try again.")

                print("\n" + "-" * 50)

            except KeyboardInterrupt:
                print("\n\n⏹️  Process interrupted by user.")
                print("👋 Thank you for using the AWS Bedrock CrewAI Data Analysis System!")
                break
            except Exception as e:
                print(f"\n❌ Error in query loop: {str(e)}")
                print("🔄 Continuing with next query...")

        print(f"\n📈 Total queries processed: {query_count}")

    except Exception as e:
        print(f"\n❌ Critical error in system initialization: {str(e)}")
        print("\nDebugging information:")
        print(f" • AWS Access Key configured: {'Yes' if aws_access_key_id else 'No'}")
        print(f" • AWS Region: {aws_region_name}")
        print(f" • LLM model: amazon.nova-micro-v1:0")
        print(f" • Embedding model: amazon.titan-embed-text-v2:0")
        print(f" • Files required: {data_dict_path}, {dataset_path}")

        # Additional debugging for AWS
        if "AWS" in str(e) or "bedrock" in str(e).lower():
            print("\n🔧 AWS Bedrock Issue Detected:")
            print(" • Check your AWS credentials and permissions")
            print(" • Ensure you have access to Amazon Nova and Titan models")
            print(" • Verify your AWS region supports these models")
            print(" • Make sure boto3 is installed: pip install boto3")

# ===============================
# MAIN EXECUTION
# ===============================

def main():
    """Main execution function - AWS Bedrock version with continuous queries"""
    continuous_query_loop()

if __name__ == "__main__":
    main()
