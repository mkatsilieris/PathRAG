import os
import sys
import time
import datetime
import traceback
import glob
import pandas as pd
import argparse
from pathlib import Path
from dotenv import load_dotenv
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import azure_openai_complete, azure_openai_embedding
from PathRAG.utils import logger

# Parse command line arguments
parser = argparse.ArgumentParser(description='PathRAG test script')
parser.add_argument('--generate-kb', action='store_true', 
                    help='Generate knowledge base from content files')
args = parser.parse_args()

# Load environment variables from .env file
load_dotenv()

# Set up more verbose logging
logger.setLevel("INFO")

# Define paths
WORKING_DIR = r"C:\Users\mkatsili\projects\PathRAG\working_repository"
CONTENT_DIR = r"C:\Users\mkatsili\projects\PathRAG\content_repository"
QA_CSV_PATH = r"C:\Users\mkatsili\projects\PathRAG\athena_qa.csv"
run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# Using a fixed Excel file name without timestamp
EXCEL_OUTPUT = r"C:\Users\mkatsili\projects\PathRAG\query_results.xlsx"

print(f"Script started at: {run_timestamp}")

# Create working directory if it doesn't exist
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR, exist_ok=True)
    print(f"Created working directory: {WORKING_DIR}")

try:
    print("Initializing PathRAG with local storage configuration...")
    
    # Initialize PathRAG with local storage options
    rag = PathRAG(
        working_dir=WORKING_DIR,
        llm_model_func=azure_openai_complete,
        embedding_func=azure_openai_embedding,
        vector_storage="NanoVectorDBStorage",
        kv_storage="JsonKVStorage",
        graph_storage="NetworkXStorage",
    )
    print("PathRAG initialized successfully!")

    if args.generate_kb:
        # Find all markdown files in the content directory
        print(f"Searching for markdown files in {CONTENT_DIR}...")
        all_md_files = []
        for root, dirs, files in os.walk(CONTENT_DIR):
            for file in files:
                if file.endswith('.md'):
                    all_md_files.append(os.path.join(root, file))
        
        print(f"Found {len(all_md_files)} markdown files to process")
        
        # Insert all files into the knowledge base
        successful_files = 0
        failed_files = 0
        
        print(f"\nProcessing files:")
        print("-" * 50)
        
        for idx, file_path in enumerate(all_md_files):
            file_name = os.path.basename(file_path)
            print(f"Processing file {idx+1}/{len(all_md_files)}: {file_name}")
            try:
                with open(file_path, encoding='utf-8') as f:
                    content = f.read()
                    print(f"  - Inserting content ({len(content)} characters)...")
                    start_time = time.time()
                    rag.insert(content)
                    end_time = time.time()
                    print(f"  - File processed successfully in {end_time - start_time:.2f} seconds")
                    successful_files += 1
            except Exception as e:
                print(f"  - ERROR processing file {file_path}: {str(e)}")
                traceback.print_exc()
                failed_files += 1
                continue
        
        print("\nFile processing summary:")
        print(f"- Successfully processed: {successful_files} files")
        print(f"- Failed to process: {failed_files} files")
        print(f"- Total files: {len(all_md_files)} files")
    else:
        print("\nSkipping knowledge base generation as requested...")
        print("Using existing knowledge base in working directory:", WORKING_DIR)
        print("Make sure the knowledge base files already exist in this directory.")

    # Read questions from athena_qa.csv
    print(f"\nReading questions from {QA_CSV_PATH}...")
    try:
        qa_df = pd.read_csv(QA_CSV_PATH, encoding='utf-8')
        print(f"Successfully loaded {len(qa_df)} questions from CSV file")
    except Exception as e:
        print(f"ERROR loading questions from CSV: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    # Define a system prompt to guide the LLM responses
    system_prompt = """You are the internal virtual assistant for the National Bank of Greece.
Your role is to support bank-related queries by delivering accurate, concise, and structured information on financial products and internal procedures.

## 🎯 Core Purpose
- Provide reliable, structured info on products and procedures.

## 📌 Core Principles
- Always reference products by name.
- Use only the given context.
- Stay focused on the query.
- Clearly outline steps and requirements.
- Use structured, easy-to-scan formats.
- Use **Markdown** for formatting and **Mermaid** for diagrams.

## 🧾 Response Types

### 📦 Product Information
- Start with a short overview.
- List related products, features, and eligibility.
- Include required documents, fees, terms, conditions.
- Mention exceptions or limits.

### 📄 Requirements
- List documents and prerequisites.
- Order steps logically.
- Explain document purposes.
- Note exceptions.

### 🔧 Procedures
- Provide step-by-step guidance.
- Show roles, timelines, and required approvals.
- Highlight exceptions.

## 📐 Formatting Standards
- Use `##`, `###` for headings.
- Bullet points and numbered lists.
- Blockquotes `>` for internal references.
- Mermaid for diagrams.
- Emoji indicators:  
  • 📊 Data  
  • 📄 Documents  
  • ⚠️ Warnings  
  • ⏱️ Timelines  
  • 🔑 Eligibility

## ⚠️ Limitations
- Acknowledge if info is missing from context.
- Never assume or invent information.
- Indicate when in-branch visits or verification are required.
"""

    # Create a list to hold our results for the Excel file
    results_for_excel = []
    
    print("\nRunning queries from CSV file:")
    print("-" * 50)
    
    # Define total number of questions to process
    total_questions = len(qa_df)
    
    # Process each question from the CSV file
    for index, row in qa_df.iterrows():
        question_id = row.get('#', f'Q{index+1}')
        question = row.get('Question', '')
        area = row.get('Περιοχή', 'Unknown')
        reference_answer = row.get('Answer', '')
        sources = row.get('Sources', '')
        
        if not question:
            print(f"Skipping row {index+1}: No question found")
            continue
        
        try:
            print(f"\nQuery {index+1}/{total_questions}: '{question}' (Area: {area})")
            start_time = time.time()
# Remove the system_prompt parameter as it's not supported
            result = rag.query(question, param=QueryParam(mode="hybrid", use_cache=False, system_prompt=system_prompt))
            end_time = time.time()
            query_time = end_time - start_time
            
            print(f"Query completed in {query_time:.2f} seconds")
            print(f"Response length: {len(str(result))} characters")
            print("Response preview: " + str(result)[:100] + "..." if len(str(result)) > 100 else str(result))
            
            # Add result to our list for Excel export with run timestamp and query ID
            results_for_excel.append({
                "Run Date": run_timestamp,
                "ID": question_id,
                "Area": area,
                "Question": question,
                "PathRAG Response": result,
                "Reference Answer": reference_answer,
                "Sources": sources,
                "Time (seconds)": round(query_time, 2)
            })
        except Exception as e:
            print(f"ERROR with query '{question}': {str(e)}")
            traceback.print_exc()
            results_for_excel.append({
                "Run Date": run_timestamp,
                "ID": question_id,
                "Area": area,
                "Question": question,
                "PathRAG Response": f"ERROR: {str(e)}",
                "Reference Answer": reference_answer,
                "Sources": sources,
                "Time (seconds)": 0
            })
    
    # Create a DataFrame from results
    if results_for_excel:
        try:
            print(f"\nSaving results to Excel file: {EXCEL_OUTPUT}")
            df_new = pd.DataFrame(results_for_excel)
            
            # Write results to Excel
            df_new.to_excel(EXCEL_OUTPUT, index=False)
            print(f"Results saved successfully to {EXCEL_OUTPUT}")
        except Exception as e:
            print(f"ERROR saving Excel file: {str(e)}")
            traceback.print_exc()
    
    print(f"\nScript completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
except Exception as e:
    print(f"\nCRITICAL ERROR: An exception occurred during execution: {str(e)}")
    traceback.print_exc()
    sys.exit(1)














