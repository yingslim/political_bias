import pandas as pd
from openai import OpenAI
import time
import logging
from tqdm import tqdm
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import shutil
from pathlib import Path

from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("API_KEY") #TODO set your API key in the .env file

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thread-safe file writing lock
file_lock = Lock()

# Initialize OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key= API_KEY 
)
    
def call_api(prompt, model, max_retries=3):
    """
    Make API call with retry logic
    """
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                extra_body={},
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return completion.choices[0].message.content
        
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"All attempts failed for prompt: {prompt[:50]}...")
                return f"ERROR: {str(e)}"

def read_csv_file(file_path):
    """
    Read regular comma-separated CSV file with proper encoding and error handling
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        logger.info(f"Successfully read CSV file: {file_path} with {len(df)} rows")
        return df
    except UnicodeDecodeError:
        # Try different encodings
        encodings = ['utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"Successfully read CSV file with {encoding} encoding: {file_path} with {len(df)} rows")
                return df
            except:
                continue
        raise ValueError(f"Could not read file {file_path} with any encoding")

def read_tab_separated_csv(file_path):
    """
    Read tab-separated CSV file with proper encoding and error handling
    """
    try:
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        logger.info(f"Successfully read tab-separated CSV file: {file_path} with {len(df)} rows")
        return df
    except UnicodeDecodeError:
        # Try different encodings
        encodings = ['utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, sep='\t', encoding=encoding)
                logger.info(f"Successfully read tab-separated CSV file with {encoding} encoding: {file_path} with {len(df)} rows")
                return df
            except:
                continue
        raise ValueError(f"Could not read file {file_path} with any encoding")

def write_tab_separated_csv(df, file_path):
    """
    Write tab-separated CSV file with proper encoding
    """
    df.to_csv(file_path, sep='\t', encoding='utf-8', index=False)

def setup_output_file(input_df, output_file_path):
    """
    Setup the output TSV file with headers and empty api_response column
    Preserves all original data and adds api_response column if missing
    """
    # Create a copy to avoid modifying original
    output_df = input_df.copy()
    
    # Add api_response column if it doesn't exist
    if 'api_response' not in output_df.columns:
        output_df['api_response'] = ''
        logger.info("Added api_response column")
    
    # Write the header and initial data to file
    write_tab_separated_csv(output_df, output_file_path)
    logger.info(f"Initialized output file: {output_file_path} with {len(output_df)} rows")
    return output_df

def create_backup(file_path):
    """
    Create a backup of the file before processing
    """
    if os.path.exists(file_path):
        backup_path = f"{file_path}.backup_{int(time.time())}"
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    return None

def update_tab_separated_csv_row_threadsafe(output_file_path, row_index, api_response):
    """
    Thread-safe update of a specific row in the tab-separated CSV file with the API response
    """
    with file_lock:
        try:
            # Read the current tab-separated CSV
            df = read_tab_separated_csv(output_file_path)
            
            # Ensure the row exists
            if row_index >= len(df):
                logger.error(f"Row index {row_index} out of range. File has {len(df)} rows.")
                return False
            
            # Update the specific row
            df.at[row_index, 'api_response'] = api_response
            
            # Write back to file
            write_tab_separated_csv(df, output_file_path)
            return True
            
        except Exception as e:
            logger.error(f"Error updating row {row_index}: {e}")
            return False

def process_single_row(args):
    """
    Process a single row - designed for multithreading
    """
    row_index, prompt, model_name, output_file_path, max_retries = args
    
    logger.info(f"Processing row {row_index}: {prompt[:50]}...")
    
    # Make API call
    response = call_api(prompt, model=model_name, max_retries=max_retries)  # Fixed: now uses model=model_name
    
    # Update the tab-separated CSV file immediately (thread-safe)
    success = update_tab_separated_csv_row_threadsafe(output_file_path, row_index, response)
    
    if success:
        logger.info(f"Row {row_index} completed and saved")
    else:
        logger.error(f"Failed to save row {row_index}")
    
    # Add small delay to avoid overwhelming the API
    time.sleep(0.1)
    
    return row_index, response, success

def get_unprocessed_rows(df, output_file_path, resume=True):
    """
    Get list of rows that need to be processed
    Ensures we maintain the exact same number of rows
    """
    rows_to_process = []
    
    if resume and os.path.exists(output_file_path):
        try:
            existing_df = read_tab_separated_csv(output_file_path)
            
            # Verify row count consistency
            if len(existing_df) != len(df):
                logger.warning(f"Row count mismatch: input {len(df)}, output {len(existing_df)}")
                # If output has fewer rows, we need to recreate it
                if len(existing_df) < len(df):
                    logger.info("Output file has fewer rows than input. Recreating...")
                    return list(range(len(df)))
            
            if 'api_response' in existing_df.columns:
                # Find rows that are empty, null, or contain errors
                for idx in range(len(existing_df)):
                    if idx < len(existing_df):
                        response = existing_df.at[idx, 'api_response']
                        if (pd.isna(response) or 
                            response == '' or 
                            str(response).startswith('ERROR:')):
                            rows_to_process.append(idx)
                
                logger.info(f"Found {len(rows_to_process)} unprocessed rows out of {len(existing_df)}")
            else:
                logger.info("No api_response column found, processing all rows")
                rows_to_process = list(range(len(df)))
                
        except Exception as e:
            logger.warning(f"Could not read existing file: {e}, processing all rows")
            rows_to_process = list(range(len(df)))
    else:
        rows_to_process = list(range(len(df)))
    
    return rows_to_process

def validate_prompt_column(df, prompt_column='generated_prompt'):
    """
    Validate that the prompt column exists and has data
    """
    if prompt_column not in df.columns:
        available_columns = list(df.columns)
        logger.error(f"Prompt column '{prompt_column}' not found. Available columns: {available_columns}")
        
        # Try to find a likely prompt column
        potential_columns = [col for col in available_columns if 'prompt' in col.lower()]
        if potential_columns:
            suggested_column = potential_columns[0]
            logger.info(f"Suggested prompt column: '{suggested_column}'")
            return suggested_column
        else:
            raise ValueError(f"No prompt column found. Please specify the correct column name.")
    
    # Check for empty prompts
    empty_prompts = df[prompt_column].isna().sum()
    if empty_prompts > 0:
        logger.warning(f"Found {empty_prompts} empty prompts")
    
    return prompt_column

def process_dataset_multithreaded(input_csv_path, output_csv_path=None, sample_size=None, 
                                resume=True, max_workers=5, rate_limit_delay=0.5,
                                prompt_column='generated_prompt', model="openai/gpt-4o-mini", max_retries=3):
    """
    Process the dataset using multithreading with robust resume capability
    Input: Regular comma-separated CSV
    Output: Tab-separated CSV
    
    Args:
        input_csv_path: Path to input CSV (comma-separated)
        output_csv_path: Path to output CSV (tab-separated, optional)
        sample_size: Number of rows to sample (optional)
        resume: Whether to resume from existing output file
        max_workers: Number of threads to use
        rate_limit_delay: Delay between API calls to respect rate limits
        prompt_column: Name of the column containing prompts
        model: Model name to use for API calls
        max_retries: Maximum retries for API calls
    """
    # Read the input CSV (comma-separated)
    df = read_csv_file(input_csv_path)
    original_row_count = len(df)
    
    # Validate prompt column
    prompt_column = validate_prompt_column(df, prompt_column)
    
    # Sample if requested (but preserve indices for consistency)
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).sort_index()
        logger.info(f"Sampled {len(df)} rows (maintaining original indices)")
    
    # Set output file path (tab-separated CSV)
    if output_csv_path is None:
        output_csv_path = input_csv_path.replace('.csv', '_with_responses.csv')
    
    # Create backup if resuming
    if resume and os.path.exists(output_csv_path):
        create_backup(output_csv_path)
    
    # Setup output file
    if not resume or not os.path.exists(output_csv_path):
        setup_df = setup_output_file(df, output_csv_path)
    else:
        # Load existing data and verify consistency
        existing_df = read_tab_separated_csv(output_csv_path)
        if len(existing_df) != original_row_count:
            logger.warning(f"Existing file has {len(existing_df)} rows, but input has {original_row_count} rows")
            logger.info("Recreating output file to maintain consistency")
            setup_df = setup_output_file(df, output_csv_path)
        else:
            setup_df = existing_df
    
    # Get rows that need processing
    rows_to_process = get_unprocessed_rows(df, output_csv_path, resume)
    
    if not rows_to_process:
        logger.info("All rows already processed!")
        final_df = read_tab_separated_csv(output_csv_path)
        return final_df
    
    # Prepare arguments for threading
    thread_args = []
    for row_idx in rows_to_process:
        if row_idx < len(df):
            prompt = df.iloc[row_idx][prompt_column]
            if pd.notna(prompt) and prompt.strip():
                thread_args.append((row_idx, prompt, model, output_csv_path, max_retries))
            else:
                logger.warning(f"Empty prompt at row {row_idx}, skipping")
                # Update with empty response to mark as processed
                update_tab_separated_csv_row_threadsafe(output_csv_path, row_idx, "EMPTY_PROMPT")
    
    if not thread_args:
        logger.info("No valid prompts to process!")
        return read_tab_separated_csv(output_csv_path)
    
    # Process with thread pool
    logger.info(f"Starting multithreaded processing with {max_workers} workers")
    logger.info(f"Processing {len(thread_args)} rows with model: {model}")
    
    # Use a semaphore to control rate limiting across threads
    rate_limiter = threading.Semaphore(max_workers)
    
    def rate_limited_process(args):
        with rate_limiter:
            result = process_single_row(args)
            time.sleep(rate_limit_delay)  # Global rate limiting
            return result
    
    completed_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(rate_limited_process, args): args for args in thread_args}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(thread_args), desc="Processing prompts") as pbar:
            for future in as_completed(future_to_args):
                try:
                    row_index, response, success = future.result()
                    if success:
                        completed_count += 1
                    else:
                        failed_count += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'completed': completed_count, 
                        'failed': failed_count,
                        'current_row': row_index
                    })
                except Exception as e:
                    args = future_to_args[future]
                    failed_count += 1
                    logger.error(f"Error processing row {args[0]}: {e}")
                    # Update with error message
                    update_tab_separated_csv_row_threadsafe(output_csv_path, args[0], f"ERROR: {str(e)}")
                    pbar.update(1)
    
    # Read the final result and verify row count
    final_df = read_tab_separated_csv(output_csv_path)
    
    # Verify we maintained the correct number of rows
    if len(final_df) != original_row_count:
        logger.error(f"Row count mismatch! Expected {original_row_count}, got {len(final_df)}")
    else:
        logger.info(f"Row count verified: {len(final_df)} rows maintained")
    
    logger.info(f"Processing complete! Results saved to {output_csv_path}")
    logger.info(f"Successfully processed {completed_count} rows, {failed_count} failed")
    
    return final_df

def check_progress(output_file_path):
    """
    Check the progress of processing
    """
    if not os.path.exists(output_file_path):
        logger.info("Output file does not exist yet")
        return
    
    try:
        df = read_tab_separated_csv(output_file_path)
        
        if 'api_response' not in df.columns:
            logger.info("No api_response column found")
            return
        
        total_rows = len(df)
        completed_rows = len(df[df['api_response'].notna() & 
                              (df['api_response'] != '') & 
                              (~df['api_response'].str.startswith('ERROR:', na=False)) &
                              (df['api_response'] != 'EMPTY_PROMPT')])
        error_rows = len(df[df['api_response'].str.startswith('ERROR:', na=False)])
        empty_rows = len(df[df['api_response'] == 'EMPTY_PROMPT'])
        pending_rows = total_rows - completed_rows - error_rows - empty_rows
        
        logger.info(f"Progress: {completed_rows}/{total_rows} completed ({completed_rows/total_rows*100:.1f}%)")
        logger.info(f"Errors: {error_rows} rows")
        logger.info(f"Empty prompts: {empty_rows} rows")
        logger.info(f"Pending: {pending_rows} rows")
        
        return {
            'total': total_rows,
            'completed': completed_rows,
            'errors': error_rows,
            'empty': empty_rows,
            'pending': pending_rows
        }
        
    except Exception as e:
        logger.error(f"Error checking progress: {e}")
        return None

def main():
    """
    Main function to easily configure and run the processing
    """
    # Input file selection
    print("\n=== CSV File Selection ===")
    input_file = input("Enter the path to your input CSV file (or press Enter for default): ").strip()
    
    # Use default if no input provided
    if not input_file:
        input_file = "./generated_prompts_multilingual_with_framing_28092025.csv"
        print(f"Using default file: {input_file}")
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found!")
        print("Please check the file path and try again.")
        return
    
    # Generate output file name
    input_file_stem = input_file.rsplit('.', 1)[0]
    output_file = f"{input_file_stem}_responses.csv"
    print(f"Output will be saved to: {output_file}")
    
    # Model selection menu
    print("\n=== Model Selection ===")
    models = {
        1: "openai/gpt-4o-mini",
        2: "qwen/qwen3-235b-a22b",
        3: "deepseek/deepseek-chat", 
        4: "meta-llama/llama-3.3-70b-instruct"
    }
    
    print("Available models:")
    for key, model in models.items():
        print(f"{key}. {model}")
    
    while True:
        try:
            choice = int(input("\nSelect a model (1-4): "))
            if choice in models:
                model_name = models[choice]
                break
            else:
                print("Invalid choice. Please select 1, 2, 3, or 4.")
        except ValueError:
            print("Invalid input. Please enter a number (1-4).")
    
    # Processing configuration
    print(f"\n=== Processing Configuration ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Selected model: {model_name}")
    
    max_workers = 15
    rate_limit_delay = 0.5
    prompt_column = 'generated_prompt'
    sample_size = None  # Set to a number if you want to test with a subset
    
    # Confirm before starting
    confirm = input("\nProceed with processing? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Processing cancelled.")
        return
    
    logger.info(f"Starting processing with model: {model_name}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    
    # Process the dataset
    result_df = process_dataset_multithreaded(
        input_csv_path=input_file,
        output_csv_path=output_file,
        sample_size=sample_size,
        resume=True,
        max_workers=max_workers,
        rate_limit_delay=rate_limit_delay,
        prompt_column=prompt_column,
        model=model_name,  # Now properly parameterized
        max_retries=3
    )
    
    logger.info("Processing completed!")
    
    # Check final progress
    check_progress(output_file)

if __name__ == "__main__":
    main()