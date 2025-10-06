import json
import pandas as pd
import numpy as np
from datetime import datetime
import random
import argparse

def load_data(file_path):
    """Load the JSON dataset from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        # Check if the data is a JSON array or a line-by-line JSON
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            # It's a JSON array
            data = json.load(f)
        else:
            # It might be line-by-line JSON
            data = [json.loads(line) for line in f if line.strip()]
    
    return data

def parse_dates(data):
    """Convert string dates to datetime objects."""
    for item in data:
        if 'date' in item:
            try:
                # Try to parse the date, assuming it's in YYYY-MM-DD format
                item['datetime'] = datetime.strptime(item['date'], '%Y-%m-%d')
            except (ValueError, TypeError):
                # If parsing fails, set datetime to None
                item['datetime'] = None
    
    # Filter out entries with invalid dates
    return [item for item in data if item.get('datetime') is not None]

def analyze_date_distribution(data):
    """Analyze the distribution of dates in the dataset."""
    # Extract dates
    dates = [item['datetime'] for item in data]
    
    # Convert to pandas Series for analysis
    date_series = pd.Series(dates)
    
    # Get basic statistics
    min_date = date_series.min()
    max_date = date_series.max()
    date_range = (max_date - min_date).days + 1
    
    # Create a histogram by month
    date_counts = date_series.dt.to_period('M').value_counts().sort_index()
    
    return {
        'min_date': min_date,
        'max_date': max_date,
        'date_range': date_range,
        'date_counts': date_counts
    }

def stratified_sampling(data, target_size=8000):
    """
    Perform stratified sampling based on date periods to maintain temporal distribution.
    
    This ensures we get a representative sample across the entire timeline.
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data)
    
    # Add a period column (by month)
    df['period'] = df['datetime'].dt.to_period('M')
    
    # Calculate the current size and sampling ratio
    current_size = len(df)
    sampling_ratio = target_size / current_size
    
    # Group by period and sample from each group
    sampled_data = []
    
    for period, group in df.groupby('period'):
        # Calculate how many samples we need from this period
        group_size = len(group)
        sample_size = max(1, int(group_size * sampling_ratio))
        
        # Sample from this period
        period_sample = group.sample(n=min(sample_size, group_size), random_state=42)
        sampled_data.append(period_sample)
    
    # Combine all sampled data
    result = pd.concat(sampled_data)
    
    # Remove added columns and convert back to list of dictionaries
    result = result.drop(columns=['datetime', 'period'])
    
    return result.to_dict('records')

def main():
    parser = argparse.ArgumentParser(description='Sample a JSON dataset across timeline.')
    parser.add_argument('input_file', help='Path to the input JSON file')
    parser.add_argument('output_file', help='Path to the output JSON file')
    parser.add_argument('--target_size', type=int, default=7000, 
                        help='Target size of the sampled dataset (default: 7000)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_data(args.input_file)
    print(f"Loaded {len(data)} entries")
    
    # Parse dates
    print("Parsing dates...")
    data_with_dates = parse_dates(data)
    print(f"Found {len(data_with_dates)} entries with valid dates")
    
    # Analyze date distribution
    print("Analyzing date distribution...")
    distribution = analyze_date_distribution(data_with_dates)
    print(f"Date range: {distribution['min_date'].date()} to {distribution['max_date'].date()} ({distribution['date_range']} days)")
    
    # Perform stratified sampling
    print(f"Performing stratified sampling to target {args.target_size} entries...")
    sampled_data = stratified_sampling(data_with_dates, args.target_size)
    print(f"Sampled dataset contains {len(sampled_data)} entries")
    
    # Save the sampled dataset
    print(f"Saving sampled dataset to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    main()