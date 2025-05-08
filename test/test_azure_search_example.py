import os
import sys
import time
import datetime
import traceback
from dotenv import load_dotenv
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import azure_openai_complete, azure_openai_embedding
from PathRAG.utils import logger

# Load environment variables from .env file
load_dotenv()

# Set up more verbose logging
logger.setLevel("INFO")

# Define paths
WORKING_DIR = "./azure_search_test_example"
run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Test document
TEST_DOCUMENT = """
# Στεγαστικό Δάνειο

Το στεγαστικό δάνειο είναι ένα δάνειο που χορηγείται από την τράπεζα για την αγορά, κατασκευή, επισκευή ή ανακαίνιση ακινήτου.

## Χαρακτηριστικά

- **Ποσό δανείου**: Έως 80% της αξίας του ακινήτου
- **Διάρκεια**: Από 5 έως 30 έτη
- **Επιτόκιο**: Κυμαινόμενο ή σταθερό
- **Εξασφάλιση**: Προσημείωση υποθήκης επί του ακινήτου

## Διαδικασία Υποβολής Αίτησης

1. Συμπλήρωση αίτησης στεγαστικού δανείου
2. Προσκόμιση απαραίτητων δικαιολογητικών
3. Αξιολόγηση αίτησης από την τράπεζα
4. Έγκριση δανείου
5. Υπογραφή σύμβασης
6. Εκταμίευση δανείου

## Απαραίτητα Δικαιολογητικά

- Ταυτότητα ή διαβατήριο
- Εκκαθαριστικά σημειώματα τελευταίων 2 ετών
- Βεβαίωση εργοδότη για μισθωτούς
- Τίτλοι ιδιοκτησίας ακινήτου
- Τοπογραφικό διάγραμμα
- Οικοδομική άδεια (για νέες κατασκευές)

## Κόστη και Έξοδα

- Έξοδα προσημείωσης
- Έξοδα εκτίμησης ακινήτου
- Έξοδα φακέλου
- Συμβολαιογραφικά έξοδα
- Ασφάλιστρα (ζωής και πυρός)

Για περισσότερες πληροφορίες, επικοινωνήστε με το κατάστημα της τράπεζας ή επισκεφθείτε την ιστοσελίδα μας.
"""

# Test query
TEST_QUERY = "Ποια είναι η διαδικασία για την υποβολή αίτησης στεγαστικού δανείου;"

def main():
    print(f"Script started at: {run_timestamp}")
    
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
        
        # Insert test document
        print("\nInserting test document...")
        start_time = time.time()
        rag.insert(TEST_DOCUMENT)
        end_time = time.time()
        print(f"Document inserted successfully in {end_time - start_time:.2f} seconds!")
        
        # Run test query
        print(f"\nRunning test query: '{TEST_QUERY}'")
        start_time = time.time()
        result = rag.query(TEST_QUERY, param=QueryParam(mode="hybrid", use_cache=False))
        end_time = time.time()
        query_time = end_time - start_time
        
        print(f"\nQuery completed in {query_time:.2f} seconds")
        print(f"Response length: {len(str(result))} characters")
        print("\nResponse:")
        print("-" * 80)
        print(result)
        print("-" * 80)
        
        # Compare with NanoVectorDBStorage
        print("\nComparing with NanoVectorDBStorage...")
        
        # Initialize PathRAG with NanoVectorDBStorage
        nano_working_dir = "./nano_vector_test_example"
        if not os.path.exists(nano_working_dir):
            os.makedirs(nano_working_dir, exist_ok=True)
        
        nano_rag = PathRAG(
            working_dir=nano_working_dir,
            llm_model_func=azure_openai_complete,
            embedding_func=azure_openai_embedding,
            vector_storage="NanoVectorDBStorage",
            kv_storage="JsonKVStorage",
            graph_storage="NetworkXStorage"
        )
        
        # Insert test document
        print("\nInserting test document into NanoVectorDBStorage...")
        start_time = time.time()
        nano_rag.insert(TEST_DOCUMENT)
        end_time = time.time()
        print(f"Document inserted successfully in {end_time - start_time:.2f} seconds!")
        
        # Run test query
        print(f"\nRunning test query with NanoVectorDBStorage: '{TEST_QUERY}'")
        start_time = time.time()
        nano_result = nano_rag.query(TEST_QUERY, param=QueryParam(mode="hybrid", use_cache=False))
        end_time = time.time()
        nano_query_time = end_time - start_time
        
        print(f"\nNanoVectorDBStorage query completed in {nano_query_time:.2f} seconds")
        print(f"Response length: {len(str(nano_result))} characters")
        print("\nNanoVectorDBStorage Response:")
        print("-" * 80)
        print(nano_result)
        print("-" * 80)
        
        # Compare performance
        print("\nPerformance Comparison:")
        print(f"Azure Search query time: {query_time:.2f} seconds")
        print(f"NanoVectorDB query time: {nano_query_time:.2f} seconds")
        print(f"Difference: {abs(query_time - nano_query_time):.2f} seconds")
        
        if query_time < nano_query_time:
            print(f"Azure Search was {(nano_query_time / query_time):.2f}x faster!")
        else:
            print(f"NanoVectorDB was {(query_time / nano_query_time):.2f}x faster!")
        
        print(f"\nScript completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: An exception occurred during execution: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
