import os
import sys
import time
import argparse
import traceback
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from PathRAG import PathRAG
from PathRAG.llm import azure_openai_complete, azure_openai_embedding
from PathRAG.utils import logger

# Set up argument parser
parser = argparse.ArgumentParser(description='Upload Markdown files to Azure Search using PathRAG')
parser.add_argument('--content-dir', type=str, default='./content_repository',
                    help='Directory containing Markdown files to upload')
parser.add_argument('--working-dir', type=str, default='./azure_search_repository',
                    help='Working directory for PathRAG')
parser.add_argument('--max-workers', type=int, default=1,
                    help='Maximum number of worker threads for parallel processing')
parser.add_argument('--recursive', action='store_true',
                    help='Recursively search for Markdown files in subdirectories')
parser.add_argument('--file-extension', type=str, default='.md',
                    help='File extension to look for (default: .md)')
parser.add_argument('--verbose', action='store_true',
                    help='Enable verbose logging')
parser.add_argument('--clean', action='store_true',
                    help='Clean the working directory before starting')

def setup_environment():
    """Load environment variables and set up logging"""
    # Load environment variables from .env file
    load_dotenv()
    
    # Set up logging
    logger.setLevel("INFO")
    
    # Verify required environment variables
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_MODEL_NAME",
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Error: The following required environment variables are missing: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or environment.")
        sys.exit(1)
    
    return {var: os.getenv(var) for var in required_vars}

def initialize_pathrag(working_dir, env_vars):
    """Initialize PathRAG with Azure Search Vector Storage"""
    print(f"Initializing PathRAG with Azure Search Vector Storage...")
    
    # Configure Azure Search parameters
    azure_search_config = {
        "endpoint": env_vars["AZURE_SEARCH_ENDPOINT"],
        "key": env_vars["AZURE_SEARCH_KEY"],
        "index_name_prefix": os.getenv("AZURE_SEARCH_INDEX_PREFIX", "pathrag")
    }
    
    # Initialize PathRAG
    rag = PathRAG(
        working_dir=working_dir,
        llm_model_func=azure_openai_complete,
        embedding_func=azure_openai_embedding,
        vector_storage="AzureSearchVectorStorage",
        kv_storage="JsonKVStorage",
        graph_storage="NetworkXStorage",
        vector_db_storage_cls_kwargs=azure_search_config
    )
    
    print("PathRAG initialized successfully!")
    return rag

def find_markdown_files(content_dir, file_extension='.md', recursive=True):
    """Find all Markdown files in the content directory"""
    print(f"Searching for {file_extension} files in {content_dir}...")
    
    content_path = Path(content_dir)
    if not content_path.exists():
        print(f"Error: Content directory {content_dir} does not exist.")
        sys.exit(1)
    
    if recursive:
        # Find all files recursively
        files = list(content_path.glob(f"**/*{file_extension}"))
    else:
        # Find only files in the top-level directory
        files = list(content_path.glob(f"*{file_extension}"))
    
    print(f"Found {len(files)} {file_extension} files.")
    return files

def process_file(file_path, rag, verbose=False):
    """Process a single file and upload it to Azure Search"""
    try:
        file_name = os.path.basename(file_path)
        if verbose:
            print(f"Processing file: {file_name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if not content.strip():
            return {"file": file_name, "status": "skipped", "reason": "empty file"}
        
        # Insert the content into PathRAG
        start_time = time.time()
        rag.insert(content)
        end_time = time.time()
        
        return {
            "file": file_name,
            "status": "success",
            "time": end_time - start_time,
            "size": len(content)
        }
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return {
            "file": file_name,
            "status": "error",
            "error": str(e)
        }

def main():
    # Parse command line arguments
    args = parser.parse_args()
    
    # Set up environment
    env_vars = setup_environment()
    
    # Create working directory if it doesn't exist
    if not os.path.exists(args.working_dir):
        os.makedirs(args.working_dir, exist_ok=True)
        print(f"Created working directory: {args.working_dir}")
    elif args.clean:
        # Clean the working directory
        import shutil
        for item in os.listdir(args.working_dir):
            item_path = os.path.join(args.working_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print(f"Cleaned working directory: {args.working_dir}")
    
    # Initialize PathRAG
    rag = initialize_pathrag(args.working_dir, env_vars)
    
    # Find Markdown files
    files = find_markdown_files(args.content_dir, args.file_extension, args.recursive)
    
    if not files:
        print(f"No {args.file_extension} files found in {args.content_dir}.")
        return
    
    # Process files
    print(f"Processing {len(files)} files...")
    
    # Statistics
    results = {
        "total": len(files),
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "start_time": time.time(),
        "details": []
    }
    
    if args.max_workers > 1:
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_file, file, rag, args.verbose): file 
                for file in files
            }
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_file), total=len(files), desc="Processing files"):
                result = future.result()
                results["details"].append(result)
                
                if result["status"] == "success":
                    results["successful"] += 1
                elif result["status"] == "error":
                    results["failed"] += 1
                elif result["status"] == "skipped":
                    results["skipped"] += 1
    else:
        # Process files sequentially
        for file in tqdm(files, desc="Processing files"):
            result = process_file(file, rag, args.verbose)
            results["details"].append(result)
            
            if result["status"] == "success":
                results["successful"] += 1
            elif result["status"] == "error":
                results["failed"] += 1
            elif result["status"] == "skipped":
                results["skipped"] += 1
    
    # Calculate total time
    results["end_time"] = time.time()
    results["total_time"] = results["end_time"] - results["start_time"]
    
    # Print summary
    print("\nUpload Summary:")
    print(f"Total files: {results['total']}")
    print(f"Successfully processed: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Skipped: {results['skipped']}")
    print(f"Total time: {results['total_time']:.2f} seconds")
    
    if results["failed"] > 0:
        print("\nFailed files:")
        for result in results["details"]:
            if result["status"] == "error":
                print(f"  - {result['file']}: {result['error']}")
    
    print("\nUpload completed!")

if __name__ == "__main__":
    main()
