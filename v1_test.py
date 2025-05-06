import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import azure_openai_complete, azure_openai_embedding


WORKING_DIR = r"C:\Users\mkatsili\projects\PathRAG\working_repository"


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Configuration for using Azure AI Search
# Uncomment and configure these lines to use Azure AI Search instead of local storage
# azure_search_config = {
#     "endpoint": "https://your-search-service-name.search.windows.net",
#     "key": "your-azure-search-admin-key",
#     "index_name_prefix": "pathrag"  # Each namespace (entities, relationships, etc.) will use this prefix
# }

rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=azure_openai_complete,
    embedding_func=azure_openai_embedding,  # Add embedding function to use Azure OpenAI
    # Uncomment to use Azure AI Search for vector storage
    # vector_storage="AzureSearchVectorStorage",  # This would need to be implemented
    # vector_db_storage_cls_kwargs=azure_search_config,
)

data_file = r"C:\Users\mkatsili\projects\PathRAG\content_repository\J495_001_2023-ΚΑΤΑΧΩΡΗΣΗ ΑΙΤΗΜΑΤΟΣ ΚΑΤΑΝΑΛΩΤΙΚΟΥ ΔΑΝΕΙΟΥ ΜΕΣΩ ΚΑΤΑΣΤΗΜΑΤΟΣ_WF.md"
question="Bηματα διαδικασιας Καταναλωτικού Δανείου"
with open(data_file, encoding='utf-8') as f:
    rag.insert(f.read())

print(rag.query(question, param=QueryParam(mode="hybrid")))














