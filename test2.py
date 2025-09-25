import os
import re
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=GEMINI_API_KEY)

# Load Excel sheets
EXCEL_FILE = "dummy_attribute_dictionary_500.xlsx"
df_sheet1 = pd.read_excel(EXCEL_FILE, sheet_name="Sheet1")
df_sheet2 = pd.read_excel(EXCEL_FILE, sheet_name="Sheet2")

# Fix for Sheet1 multi-row keys: Assume first two columns are key and description
df_sheet1 = df_sheet1.iloc[:, :2]  # Take first two columns if unnamed
df_sheet1.columns = ["Key", "Description"]  # Rename for consistency
df_sheet1["Key"] = df_sheet1[
    "Key"
].ffill()  # Fill forward keys for multi-row descriptions
df_sheet1["Description"] = df_sheet1["Description"].fillna(
    ""
)  # Handle NaN descriptions
# Group by key and join descriptions with newlines for readability
grouped_sheet1 = df_sheet1.groupby("Key")["Description"].apply("\n".join).reset_index()

# Create documents for vector DB
documents = []
metadatas = []
ids = []

# Sheet1 documents (from grouped)
for idx, row in grouped_sheet1.iterrows():
    doc = f"Key: {row['Key']}. Description: {row['Description']}."
    documents.append(doc)
    metadatas.append({"sheet": "Sheet1", "key": row["Key"]})
    ids.append(f"sheet1_{idx}")

# Sheet2 documents
for idx, row in df_sheet2.iterrows():
    doc = (
        f"Attribute: {row['NAME']}. Description: {row['DESCRIPTION']}. "
        f"Units: {row['UNITS']}. Length: {row['LENGTH']}. "
        f"Valid Values: {row['VALID VALUES']}. Attribute Version: {row['ATTRIBUTE VERSION']}. "
        f"Default Values & Definitions: {row['DEFAULT VALUES & DEFINITIONS']}."
    )
    documents.append(doc)
    metadatas.append({"sheet": "Sheet2", "key": row["NAME"]})
    ids.append(f"sheet2_{idx}")

# Initialize embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = embedder.encode(documents).tolist()

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./attribute_vector_db")
collection = client.get_or_create_collection(name="attributes")

# Populate if empty
if collection.count() == 0:
    collection.add(
        documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids
    )
    print("Vector DB created and populated.")
else:
    print("Vector DB already exists.")


# Query function for vector search
def query_vector_db(query_text, top_k=3):
    query_embedding = embedder.encode([query_text])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results


# Generate response with Gemini
def generate_response_with_gemini(user_query, retrieved_results):
    if not retrieved_results["documents"] or not retrieved_results["documents"][0]:
        return "No relevant information found."

    context = []
    for i in range(len(retrieved_results["documents"][0])):
        doc = retrieved_results["documents"][0][i]
        meta = retrieved_results["metadatas"][0][i]
        dist = retrieved_results["distances"][0][i]
        context.append(f"Source (key: {meta['key']}, distance: {dist:.2f}): {doc}")

    context_str = "\n\n".join(context)

    prompt = f"""You are an expert on financial attribute dictionaries.
Based only on this context:
{context_str}

Answer: {user_query}

Be clear and concise. Summarize key details like description, units, etc. Suggest closest if no exact match."""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


# Interaction loop
print(
    "Welcome! Ask about attributes (e.g., 'meaning of P13_CRD1409_1'). Type 'exit' to quit."
)
while True:
    user_query = input("Your query: ")
    if user_query.lower() == "exit":
        print("Goodbye!")
        break

    # Extract potential attribute code from query
    code_match = re.search(r"P13_\w+_\d+", user_query)
    exact_results = None
    if code_match:
        code = code_match.group()
        exact_results = collection.get(where={"key": code})

    if exact_results and exact_results["documents"]:
        # Format as query results (no distances)
        results = {
            "documents": [exact_results["documents"]],
            "metadatas": [exact_results["metadatas"]],
            "distances": [
                [0.0] * len(exact_results["documents"])
            ],  # Fake perfect match
        }
    else:
        results = query_vector_db(user_query)

    response = generate_response_with_gemini(user_query, results)
    print("\nResponse:")
    print(response)
    print("\n---")












    ```python
import os
import re
import pandas as pd
import chromadb
import boto3
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Assume AWS credentials are configured via .env or IAM role: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION

def bedrock_embed(texts, model_id='amazon.titan-embed-text-v2:0'):
    client = boto3.client('bedrock-runtime', region_name=os.getenv('AWS_REGION', 'us-east-1'))
    embeddings = []
    for text in texts:
        body = json.dumps({
            "inputText": text,
            "dimensions": 384,  # Adjust as needed (256, 512, 1024)
            "normalize": True
        })
        response = client.invoke_model(
            body=body,
            modelId=model_id,
            accept='application/json',
            contentType='application/json'
        )
        response_body = json.loads(response['body'].read())
        embeddings.append(response_body['embedding'])
    return embeddings

def bedrock_generate(prompt, model_id='amazon.nova-micro_v1:0'):
    client = boto3.client('bedrock-runtime', region_name=os.getenv('AWS_REGION', 'us-east-1'))
    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 4096,
            "stopSequences": [],
            "temperature": 0.7,
            "topP": 1
        }
    })
    response = client.invoke_model(
        body=body,
        modelId=model_id,
        accept='application/json',
        contentType='application/json'
    )
    response_body = json.loads(response['body'].read())
    return response_body['outputText']

# Load Excel sheets
EXCEL_FILE = "dummy_attribute_dictionary_500.xlsx"
df_sheet1 = pd.read_excel(EXCEL_FILE, sheet_name="Sheet1")
df_sheet2 = pd.read_excel(EXCEL_FILE, sheet_name="Sheet2")

# Fix for Sheet1 multi-row keys: Assume first two columns are key and description
df_sheet1 = df_sheet1.iloc[:, :2]  # Take first two columns if unnamed
df_sheet1.columns = ['Key', 'Description']  # Rename for consistency
df_sheet1['Key'] = df_sheet1['Key'].ffill()  # Fill forward keys for multi-row descriptions
df_sheet1['Description'] = df_sheet1['Description'].fillna('')  # Handle NaN descriptions
# Group by key and join descriptions with newlines for readability
grouped_sheet1 = df_sheet1.groupby('Key')['Description'].apply('\n'.join).reset_index()

# Create documents for vector DB
documents = []
metadatas = []
ids = []

# Sheet1 documents (from grouped)
for idx, row in grouped_sheet1.iterrows():
    doc = f"Key: {row['Key']}. Description: {row['Description']}."
    documents.append(doc)
    metadatas.append({"sheet": "Sheet1", "key": row["Key"]})
    ids.append(f"sheet1_{idx}")

# Sheet2 documents
for idx, row in df_sheet2.iterrows():
    doc = (
        f"Attribute: {row['NAME']}. Description: {row['DESCRIPTION']}. "
        f"Units: {row['UNITS']}. Length: {row['LENGTH']}. "
        f"Valid Values: {row['VALID VALUES']}. Attribute Version: {row['ATTRIBUTE VERSION']}. "
        f"Default Values & Definitions: {row['DEFAULT VALUES & DEFINITIONS']}."
    )
    documents.append(doc)
    metadatas.append({"sheet": "Sheet2", "key": row["NAME"]})
    ids.append(f"sheet2_{idx}")

# Generate embeddings using Bedrock
embeddings = bedrock_embed(documents)

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./attribute_vector_db")
collection = client.get_or_create_collection(name="attributes")

# Populate if empty
if collection.count() == 0:
    collection.add(
        documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids
    )
    print("Vector DB created and populated.")
else:
    print("Vector DB already exists.")


# Query function for vector search
def query_vector_db(query_text, top_k=3):
    query_embedding = bedrock_embed([query_text])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results


# Generate response with Bedrock
def generate_response_with_bedrock(user_query, retrieved_results):
    if not retrieved_results["documents"] or not retrieved_results["documents"][0]:
        return "No relevant information found."

    context = []
    for i in range(len(retrieved_results["documents"][0])):
        doc = retrieved_results["documents"][0][i]
        meta = retrieved_results["metadatas"][0][i]
        dist = retrieved_results["distances"][0][i]
        context.append(f"Source (key: {meta['key']}, distance: {dist:.2f}): {doc}")

    context_str = "\n\n".join(context)

    prompt = f"""You are an expert on financial attribute dictionaries.
Based only on this context:
{context_str}

Answer: {user_query}

Be clear and concise. Summarize key details like description, units, etc. Suggest closest if no exact match."""

    response = bedrock_generate(prompt)
    return response


# Interaction loop
print(
    "Welcome! Ask about attributes (e.g., 'meaning of P13_CRD1409_1'). Type 'exit' to quit."
)
while True:
    user_query = input("Your query: ")
    if user_query.lower() == "exit":
        print("Goodbye!")
        break

    # Extract potential attribute code from query
    code_match = re.search(r"P13_\w+_\d+", user_query)
    exact_results = None
    if code_match:
        code = code_match.group()
        exact_results = collection.get(where={"key": code})

    if exact_results and exact_results["documents"]:
        # Format as query results (no distances)
        results = {
            "documents": [exact_results["documents"]],
            "metadatas": [exact_results["metadatas"]],
            "distances": [
                [0.0] * len(exact_results["documents"])
            ],  # Fake perfect match
        }
    else:
        results = query_vector_db(user_query)

    response = generate_response_with_bedrock(user_query, results)
    print("\nResponse:")
    print(response)
    print("\n---")
```
