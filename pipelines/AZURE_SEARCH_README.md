# Azure Search Integration for PathRAG

This document explains how to use Azure AI Search with PathRAG for vector storage and retrieval.

## Prerequisites

1. An Azure account with an active subscription
2. An Azure AI Search service
3. An Azure OpenAI service
4. Python 3.8 or later

## Setup

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file with the following environment variables:

```
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-openai-service.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_MODEL_NAME=gpt-4o-global-standard

# Azure AI Search Configuration
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_KEY=your_search_admin_key
AZURE_SEARCH_INDEX_PREFIX=pathrag
```

## Uploading Content to Azure Search

The `upload_content_to_azure.py` script allows you to upload Markdown files to Azure Search using PathRAG.

### Usage

```bash
python upload_content_to_azure.py --content-dir ./your_content_directory
```

### Command Line Arguments

- `--content-dir`: Directory containing Markdown files to upload (default: ./content_repository)
- `--working-dir`: Working directory for PathRAG (default: ./azure_search_repository)
- `--max-workers`: Maximum number of worker threads for parallel processing (default: 1)
- `--recursive`: Recursively search for Markdown files in subdirectories
- `--file-extension`: File extension to look for (default: .md)
- `--verbose`: Enable verbose logging
- `--clean`: Clean the working directory before starting

### Examples

Upload all Markdown files from a directory:

```bash
python upload_content_to_azure.py --content-dir ./docs
```

Upload files recursively with parallel processing:

```bash
python upload_content_to_azure.py --content-dir ./docs --recursive --max-workers 4
```

Upload files with a different extension:

```bash
python upload_content_to_azure.py --content-dir ./docs --file-extension .txt
```

## Using Azure Search in PathRAG

To use Azure Search in your own PathRAG applications, initialize PathRAG with the following configuration:

```python
from PathRAG import PathRAG
from PathRAG.llm import azure_openai_complete, azure_openai_embedding

# Configure Azure Search parameters
azure_search_config = {
    "endpoint": os.getenv("AZURE_SEARCH_ENDPOINT"),
    "key": os.getenv("AZURE_SEARCH_KEY"),
    "index_name_prefix": os.getenv("AZURE_SEARCH_INDEX_PREFIX", "pathrag")
}

# Initialize PathRAG with Azure Search Vector Storage
rag = PathRAG(
    working_dir="./your_working_dir",
    llm_model_func=azure_openai_complete,
    embedding_func=azure_openai_embedding,
    vector_storage="AzureSearchVectorStorage",
    kv_storage="JsonKVStorage",
    graph_storage="NetworkXStorage",
    vector_db_storage_cls_kwargs=azure_search_config
)
```

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**: Ensure all required environment variables are set in your `.env` file.

2. **Permission Issues**: Make sure your Azure Search service has the necessary permissions and your API key has admin rights.

3. **Index Limitations**: Azure Search has limits on index size and document count. Check the [Azure documentation](https://learn.microsoft.com/en-us/azure/search/search-limits-quotas-capacity) for details.

4. **Rate Limiting**: If you're processing a large number of files, you might hit rate limits. Try reducing the `--max-workers` value.

### Getting Help

If you encounter issues, check the following:

1. Azure Search service logs in the Azure portal
2. PathRAG logs (set `--verbose` flag for more detailed logs)
3. Azure OpenAI service logs

## Additional Resources

- [Azure AI Search Documentation](https://learn.microsoft.com/en-us/azure/search/)
- [Azure OpenAI Service Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [PathRAG Documentation](https://github.com/pathrag/pathrag)
