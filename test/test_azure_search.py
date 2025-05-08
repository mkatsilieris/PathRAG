import os
import sys
import time
from dotenv import load_dotenv
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import azure_openai_complete, azure_openai_embedding
from PathRAG.utils import logger

# Load environment variables from .env file
load_dotenv()

# Set up more verbose logging
logger.setLevel("INFO")

# Define paths
WORKING_DIR = "./azure_search_test"

print(f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Create working directory if it doesn't exist
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR, exist_ok=True)
    print(f"Created working directory: {WORKING_DIR}")

try:
    print("Initializing PathRAG with Azure Search Vector Storage...")
    
    # Configure Azure Search parameters
    azure_search_config = {
        "endpoint": os.getenv("AZURE_SEARCH_ENDPOINT"),
        "key": os.getenv("AZURE_SEARCH_KEY"),
        "index_name_prefix": os.getenv("AZURE_SEARCH_INDEX_PREFIX", "pathrag")
    }
    
    # Initialize PathRAG with Azure Search Vector Storage
    rag = PathRAG(
        working_dir=WORKING_DIR,
        llm_model_func=azure_openai_complete,
        embedding_func=azure_openai_embedding,
        vector_storage="AzureSearchVectorStorage",
        kv_storage="JsonKVStorage",
        graph_storage="NetworkXStorage",
        vector_db_storage_cls_kwargs=azure_search_config
    )
    print("PathRAG initialized successfully!")

    # Test with a simple document
    test_document = """
    Azure AI Search (formerly Azure Cognitive Search) is a cloud search service that gives developers APIs and tools for building rich search experiences over private, heterogeneous content in web, mobile, and enterprise applications.
    
    A search service is created in a resource group, along with other Azure resources that compose your solution. The search service itself consists of the following elements:
    
    A search engine for information storage, processing, and retrieval
    Built-in capabilities that enhance, enrich, and transform content during indexing
    Rich query syntax for text search, fuzzy search, autocomplete, geo-search and more
    Programmable through REST APIs and client libraries in Azure SDKs
    """
    
    print("Inserting test document...")
    rag.insert(test_document)
    print("Document inserted successfully!")
    
    # Test querying
    test_query = "What is Azure AI Search?"
    print(f"Testing query: '{test_query}'")
    
    result = rag.query(test_query, param=QueryParam(mode="hybrid", use_cache=False))
    
    print("Query result:")
    print(result)
    
    print("Test completed successfully!")
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
