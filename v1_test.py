import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import azure_openai_complete


WORKING_DIR = r"C:\Users\mkatsili\projects\PathRAG\content_repository"

# api_key=""
# os.environ["OPENAI_API_KEY"] = api_key
# base_url="https://api.openai.com/v1"
# os.environ["OPENAI_API_BASE"]=base_url


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=azure_openai_complete,  
)

data_file = r"C:\Users\mkatsili\projects\PathRAG\content_repository\J495_001_2023-ΚΑΤΑΧΩΡΗΣΗ ΑΙΤΗΜΑΤΟΣ ΚΑΤΑΝΑΛΩΤΙΚΟΥ ΔΑΝΕΙΟΥ ΜΕΣΩ ΚΑΤΑΣΤΗΜΑΤΟΣ_WF.md"
question="βηματα διαδικασιας Καταναλωτικού Δανείου"
with open(data_file, encoding='utf-8') as f:
    rag.insert(f.read())

print(rag.query(question, param=QueryParam(mode="hybrid")))














