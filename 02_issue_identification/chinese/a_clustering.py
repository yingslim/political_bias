#!/usr/bin/env python3
import json
import os
import glob
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from collections import Counter
import csv

# Create output directory if it doesn't exist
os.makedirs("output_chinese", exist_ok=True)
os.makedirs("output_chinese/clusters", exist_ok=True)

# Load all JSON files from directory
def load_all_json_files(directory):
    all_items = []
    file_count = 0
    
    # Find all JSON files in the directory
    json_files = glob.glob(os.path.join(directory, "*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Handle both single articles and lists of articles
                if isinstance(data, list):
                    all_items.extend(data)
                else:
                    all_items.append(data)
                    
                file_count += 1
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {file_count} JSON files containing {len(all_items)} articles")
    return all_items

# Function to extract text content from articles with different field names
def extract_content(items):
    contents = []
    titles = []
    
    for item in items:
        # Extract title
        title = item.get("title", "No Title")
        titles.append(title)
        
        # Extract content based on available fields
        if "text" in item:
            contents.append(item["text"])
        elif "content" in item:
            contents.append(item["content"])
        elif "body_content" in item:
            contents.append(item["body_content"])
        else:
            contents.append("")
    
    return contents, titles

# Function to perform clustering on embeddings
def cluster_articles(embeddings, min_cluster_size=15, n_components=20):
    # Dimensionality reduction with UMAP
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Cluster with HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(reduced_embeddings)
    
    return labels


# Save articles by cluster with metadata
def save_articles_by_cluster(items, labels, output_file="output_chinese/articles_with_clusters.json"):
    # Assign cluster labels to articles
    for item, label in zip(items, labels):
        item["cluster"] = int(label)
    
    # Save the full dataset with cluster assignments
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    
    # Also save a CSV with basic article info and cluster assignments
    df = pd.DataFrame([{
        'title': item.get('title', 'No Title'),
        'date': item.get('date', item.get('published_date', 'Unknown')),
        'cluster': item.get('cluster', -1)
    } for item in items])
    
    df.to_csv("output_chinese/article_clusters_summary.csv", index=False)
    
    # Count cluster sizes
    cluster_counts = Counter(labels)
    print("\nCluster sizes:")
    for lbl, count in cluster_counts.most_common():
        print(f"  {lbl:>3}: {count}")
    
    # Save cluster statistics
    with open("output_chinese/cluster_stats.txt", "w") as f:
        f.write("Cluster sizes:\n")
        for lbl, count in cluster_counts.most_common():
            f.write(f"  {lbl:>3}: {count}\n")
        
        f.write(f"\nTotal articles: {len(items)}\n")
        f.write(f"Clustered articles: {len(items) - cluster_counts.get(-1, 0)}\n")
        f.write(f"Unclustered articles (noise): {cluster_counts.get(-1, 0)}\n")

# Main execution
if __name__ == "__main__":
    # Ask for input directory
    json_dir = 'newspaper_articles/china' # news articles on china are in english
    
    # Load all JSON files
    all_items = load_all_json_files(json_dir)
    
    if not all_items:
        print("No articles found. Exiting.")
        exit(1)
    
    # Extract content for embedding
    contents, titles = extract_content(all_items)
    print(f"Extracted content from {len(contents)} articles")
    
    # # Generate embeddings
    print("Generating embeddings (this may take a while)...")
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2') #'jinaai/jina-embeddings-v2-base-zh', #sentence-transformers/distiluse-base-multilingual-cased-v2 #"all-mpnet-base-v2"
    embeddings = model.encode(contents, show_progress_bar=True, convert_to_numpy=True)
    # model = SentenceTransformer('jinaai/jina-embeddings-v2-base-zh', trust_remote_code=True)
    # embeddings = model.encode(contents, show_progress_bar=True, convert_to_numpy=True, batch_size=32)

    # Ask for clustering parameters
    min_cluster_size = int(input("Enter minimum cluster size (default: 15): ") or 15)
    n_components = int(input("Enter UMAP dimensions (default: 20): ") or 20)
    
    # Perform clustering
    print("Clustering articles...")
    labels = cluster_articles(embeddings, min_cluster_size, n_components)
    
    # Save articles with cluster assignments
    save_articles_by_cluster(all_items, labels)
    
    
    print("\nProcess completed successfully!")
    print("Summary:")
    print(f"- Articles loaded: {len(all_items)}")
    print(f"- Clusters found: {len(set(labels) - {-1})}")
    print(f"- Full data with clusters saved to output_chinese/articles_with_clusters.json")
    print(f"- Summary CSV saved to output_chinese/article_clusters_summary.csv")
    print(f"- Cluster statistics saved to output_chinese/cluster_stats.txt")