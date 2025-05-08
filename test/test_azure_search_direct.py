import os
import sys
import time
import asyncio
import numpy as np
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    VectorSearchAlgorithmConfiguration,
)
from openai import AsyncAzureOpenAI

# Load environment variables from .env file
load_dotenv()

# Azure Search configuration
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_PREFIX = os.getenv("AZURE_SEARCH_INDEX_PREFIX", "pathrag")
INDEX_NAME = f"{AZURE_SEARCH_INDEX_PREFIX}-test"

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Test document
TEST_DOCUMENT = """
Azure AI Search (formerly Azure Cognitive Search) is a cloud search service that gives developers APIs and tools for building rich search experiences over private, heterogeneous content in web, mobile, and enterprise applications.

A search service is created in a resource group, along with other Azure resources that compose your solution. The search service itself consists of the following elements:

A search engine for information storage, processing, and retrieval
Built-in capabilities that enhance, enrich, and transform content during indexing
Rich query syntax for text search, fuzzy search, autocomplete, geo-search and more
Programmable through REST APIs and client libraries in Azure SDKs
"""

async def generate_embeddings(texts):
    """Generate embeddings using Azure OpenAI"""
    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    response = await openai_async_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
        encoding_format="float"
    )

    return np.array([dp.embedding for dp in response.data])

def create_or_update_search_index():
    """Create or update the Azure Search index with vector search capabilities"""
    # Initialize the search index client
    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )

    # Check if index exists and delete it
    try:
        index_client.delete_index(INDEX_NAME)
        print(f"Deleted existing index: {INDEX_NAME}")
    except Exception as e:
        print(f"Index doesn't exist or couldn't be deleted: {e}")

    # Define fields for the index
    fields = [
        SearchField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
        SearchField(
            name="vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=1536,  # Dimension for text-embedding-3-small
            vector_search_profile_name="default-profile"
        ),
    ]

    # Configure vector search using raw dictionary to bypass SDK limitations
    vector_search = {
        "profiles": [
            {
                "name": "default-profile",
                "algorithm": "default-config"
            }
        ],
        "algorithms": [
            {
                "name": "default-config",
                "kind": "hnsw",
                "hnswParameters": {
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine"
                }
            }
        ]
    }

    # Create the index with vector search
    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search
    )

    # Create the index
    result = index_client.create_or_update_index(index)
    print(f"Created index {INDEX_NAME} with vector search capabilities")
    return result

async def upload_document():
    """Upload a test document with vector embeddings"""
    # Generate embedding for the document
    embeddings = await generate_embeddings([TEST_DOCUMENT])
    embedding = embeddings[0]

    # Initialize the search client
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )

    # Create document with embedding
    document = {
        "id": "test-doc-1",
        "content": TEST_DOCUMENT,
        "vector": embedding.tolist()
    }

    # Upload document
    result = search_client.upload_documents(documents=[document])
    print(f"Uploaded document: {result[0].succeeded}")
    return result

async def search_document(query_text):
    """Search for documents using vector search"""
    # Generate embedding for the query
    embeddings = await generate_embeddings([query_text])
    embedding = embeddings[0]

    # Initialize the search client
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )

    # Perform vector search using raw request
    search_request = {
        "search": "",  # Empty search text
        "select": "id,content",
        "top": 5,
        "vectorQueries": [
            {
                "vector": embedding.tolist(),
                "fields": "vector",
                "k": 5
            }
        ]
    }

    # Use the search_post method directly
    results = search_client._client.documents.search_post(search_request=search_request)

    # Print results
    print("\nSearch results:")

    # Extract results from the response
    if hasattr(results, 'value') and results.value:
        for result in results.value:
            print(f"ID: {result.get('id', 'N/A')}")
            print(f"Content: {result.get('content', '')[:100]}...")
            if '@search.score' in result:
                print(f"Score: {result.get('@search.score', 0)}")
            print("-" * 50)
    else:
        print("No results found.")

    return results

async def main():
    print("Starting Azure Search Vector Storage test...")

    # Create or update the search index
    create_or_update_search_index()

    # Wait for the index to be ready
    print("Waiting for index to be ready...")
    time.sleep(5)

    # Upload test document
    await upload_document()

    # Wait for the document to be indexed
    print("Waiting for document to be indexed...")
    time.sleep(5)

    # Search for documents
    await search_document("What is Azure AI Search?")

    print("\nTest completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
