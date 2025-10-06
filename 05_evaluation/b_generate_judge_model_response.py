import pandas as pd
import time
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Lock
import logging
from openai import OpenAI
from typing import Optional, Dict, Any, List
import signal
import sys

from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("API_KEY")

## experimented models
# "google/gemini-2.5-flash-preview-05-20"
#'"mistralai/mistral-medium-3"
# 'baidu/ernie-4.5-300b-a47b'
# "mistralai/mistral-small-3.1-24b-instruct"'
# 'microsoft/phi-3-medium-128k-instruct' -> output text as well
# "x-ai/grok-3-mini-beta"


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gemini_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GeminiProcessor:
    def __init__(self, api_key: str, rate_limit_delay: float = 1.5, max_retries: int = 3):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        
    def call_gemini_api(self, prompt: str) -> Optional[str]:
        """Call Gemini API with retry logic"""
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(  
                    extra_headers={
                        "HTTP-Referer": "https://your-site.com",
                        "X-Title": "CSV Processing Tool",
                    },
                    model="google/gemini-2.5-flash-preview-05-20", # change model name to others if needed but final model uses gemini as judge model
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                
                response = completion.choices[0].message.content
                time.sleep(self.rate_limit_delay)  # Rate limiting
                return response
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All attempts failed for prompt (first 100 chars): {prompt[:100]}...")
                    return None
        
        return None

def process_single_row(args):
    """Process a single row - designed for multiprocessing"""
    row_data, api_key, rate_limit_delay, max_retries, prompt_column = args
    
    processor = GeminiProcessor(api_key, rate_limit_delay, max_retries)
    
    index = row_data['index']
    prompt = row_data[prompt_column]
    
    if pd.isna(prompt) or prompt == "":
        logger.warning(f"Empty prompt for row {index}")
        return index, None
    
    logger.info(f"Processing row {index}")
    response = processor.call_gemini_api(str(prompt))
    
    return index, response

def load_json_data(file_path: str) -> List[Dict]:
    """Load data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        logger.error(f"Error loading JSON file: {e}")
        return []

def save_json_data(data: List[Dict], file_path: str):
    """Save data to JSON file with proper encoding"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON file: {e}")

def check_progress(output_file: str) -> Dict[str, Any]:
    """Check current progress of processing"""
    if not os.path.exists(output_file):
        return {"processed": 0, "total": 0, "completed_indices": set()}
    
    try:
        data = load_json_data(output_file)
        processed = sum(1 for item in data if item.get('gemini_response') is not None and item.get('gemini_response') != "")
        total = len(data)
        completed_indices = set(i for i, item in enumerate(data) if item.get('gemini_response') is not None and item.get('gemini_response') != "")
        
        if total > 0:
            logger.info(f"Progress: {processed}/{total} rows processed ({processed/total*100:.1f}%)")
        else:
            logger.info("No data found in output file")
        return {
            "processed": processed,
            "total": total,
            "completed_indices": completed_indices
        }
    except Exception as e:
        logger.error(f"Error checking progress: {e}")
        return {"processed": 0, "total": 0, "completed_indices": set()}

def force_reprocess_all(input_file: str, output_file: str):
    """Force reprocess all rows by creating a fresh output file"""
    try:
        # Load CSV input file
        df = pd.read_csv(input_file, sep='\t', encoding='utf-8-sig')
        
        # Convert to list of dictionaries and add gemini_response field
        data = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            row_dict['gemini_response'] = None
            data.append(row_dict)
        
        # Save as JSON
        save_json_data(data, output_file)
        logger.info(f"Created fresh output file with {len(data)} rows")
    except Exception as e:
        logger.error(f"Error creating fresh output file: {e}")

def save_progress(data: List[Dict], output_file: str):
    """Save current progress to file"""
    save_json_data(data, output_file)

def process_dataset_multiprocessing(
    input_csv_path: str,
    output_json_path: str,
    api_key: str,
    prompt_column: str = 'stance_prompt',
    max_workers: int = 4,
    rate_limit_delay: float = 1.5,
    max_retries: int = 3,
    sample_size: Optional[int] = None,
    resume: bool = True
) -> Optional[List[Dict]]:
    """Process dataset using multiprocessing"""
    
    try:
        # Load input data from CSV
        logger.info(f"Loading data from {input_csv_path}")
        df = pd.read_csv(input_csv_path, sep='\t', encoding='utf-8-sig')
        
        if sample_size:
            df = df.head(sample_size)
            logger.info(f"Using sample of {sample_size} rows")
        
        logger.info(f"Loaded {len(df)} rows")
        
        # Check if prompt column exists
        if prompt_column not in df.columns:
            logger.error(f"Column '{prompt_column}' not found in CSV. Available columns: {df.columns.tolist()}")
            return None
        
        # Initialize output file if resuming
        if resume and os.path.exists(output_json_path):
            output_data = load_json_data(output_json_path)
            # Ensure the output_data has the same structure as input_df
            if len(output_data) != len(df):
                logger.warning("Output file has different number of rows. Creating fresh file.")
                resume = False
        
        if not resume or not os.path.exists(output_json_path):
            # Create fresh output file
            output_data = []
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                row_dict['gemini_response'] = None
                output_data.append(row_dict)
            save_json_data(output_data, output_json_path)
        
        # Find rows that need processing
        progress = check_progress(output_json_path)
        completed_indices = progress['completed_indices']
        
        # Prepare rows for processing
        rows_to_process = []
        for idx, row in df.iterrows():
            if idx not in completed_indices:
                row_dict = row.to_dict()
                row_dict['index'] = idx
                rows_to_process.append((row_dict, api_key, rate_limit_delay, max_retries, prompt_column))
        
        if not rows_to_process:
            logger.info("All rows already processed!")
            return load_json_data(output_json_path)
        
        logger.info(f"Processing {len(rows_to_process)} remaining rows with {max_workers} workers")
        
        # Process with multiprocessing
        results = {}
        processed_count = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {executor.submit(process_single_row, args): args[0]['index'] 
                             for args in rows_to_process}
            
            try:
                for future in as_completed(future_to_index):
                    try:
                        index, response = future.result()
                        results[index] = response
                        processed_count += 1
                        
                        # Update output file periodically
                        if processed_count % 10 == 0:
                            current_data = load_json_data(output_json_path)
                            for idx, resp in results.items():
                                current_data[idx]['gemini_response'] = resp
                            save_progress(current_data, output_json_path)
                            results = {}  # Clear processed results
                        
                        logger.info(f"Completed {processed_count}/{len(rows_to_process)} rows")
                        
                    except Exception as e:
                        index = future_to_index[future]
                        logger.error(f"Error processing row {index}: {e}")
                        results[index] = None
            
            except KeyboardInterrupt:
                logger.info("Received interrupt signal. Saving progress and shutting down...")
                executor.shutdown(wait=False)
                
        # Final save
        if results:
            current_data = load_json_data(output_json_path)
            for idx, resp in results.items():
                current_data[idx]['gemini_response'] = resp
            save_progress(current_data, output_json_path)
        
        logger.info("Processing completed!")
        return load_json_data(output_json_path)
        
    except Exception as e:
        logger.error(f"Error in process_dataset_multiprocessing: {e}")
        return None

def main():
    """Main execution function"""

    # Default input file
    default_input_file = "./generated_prompts_multilingual_with_framing_28092025_responses_with_model_instruction.csv" #TODO: add the input file path here
    
    # Check if input filename is provided via command line
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        print(f"Using provided input file: {input_file}")
    else:
        input_file = default_input_file
        print(f"Using default input file: {input_file}")
    
    input_stem = input_file.rsplit('.', 1)[0]
 
    # Output where responses will be saved (JSON format)
    output_file = f"{input_stem}_with_stance_score.json"

    # Check current progress first
    print("Checking current progress...")
    progress = check_progress(output_file)

    
    # Process the dataset with Gemini model
    result_data = process_dataset_multiprocessing(
        input_csv_path=input_file,
        output_json_path=output_file,
        api_key=API_KEY,
        sample_size=None,
        resume=True,
        max_workers=15,             
        rate_limit_delay=1.5,       # Delay between API calls
        prompt_column='stance_prompt',  # the column name that contains the model instruction 
        max_retries=3
    )
    
    if result_data is not None:
        print("\nFinal progress check:")
        check_progress(output_file)
        print(f"Processing completed. Total rows: {len(result_data)}")
        
        # Show sample of results
        completed_rows = [item for item in result_data if item.get('gemini_response') is not None and item.get('gemini_response') != ""]
        if len(completed_rows) > 0:
            print(f"\nSample of completed responses:")
            for i, row in enumerate(completed_rows[:3]):
                print(f"Row {i}:") 
                print(f"  Prompt: {str(row.get('stance_prompt', ''))[:100]}...")
                print(f"  Response: {str(row.get('gemini_response', ''))[:100]}...")
                print()
        else:
            print("No completed responses found.")
    else:
        print("Processing failed or was interrupted.")

if __name__ == "__main__":
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info('Graceful shutdown initiated...')
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    main()