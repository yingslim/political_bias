#!/usr/bin/env python3
import json
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import csv
import re

def clean_text(text):
    """
    Clean text to remove URLs, special characters, and extra whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove special characters but keep spaces between words
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_tfidf_for_clusters(input_file, csv_clusters_file=None, max_features=100, min_df=2):
    """
    Generate TF-IDF for each cluster in the input file
    
    Args:
        input_file: Path to JSON file with articles and cluster assignments
        csv_clusters_file: Optional path to CSV file with cluster assignments
        max_features: Maximum number of features to extract per cluster
        min_df: Minimum document frequency (to filter out rare terms)
    """
    print(f"Loading articles from {input_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs("output_english/tfidf_clusters", exist_ok=True)
    
    # Load the JSON data
    with open(input_file, "r", encoding="utf-8") as f:
        articles = json.load(f)
    
    print(f"Loaded {len(articles)} articles")
    
    # If CSV file is provided, use it to update cluster assignments
    if csv_clusters_file and os.path.exists(csv_clusters_file):
        print(f"Loading cluster assignments from {csv_clusters_file}")
        df_clusters = pd.read_csv(csv_clusters_file)
        
        # Create a mapping from title to cluster
        title_to_cluster = dict(zip(df_clusters['title'], df_clusters['cluster']))
        
        # Update cluster assignments in the articles
        for article in articles:
            title = article.get('title', '')
            if title in title_to_cluster:
                article['cluster'] = title_to_cluster[title]
    
    # Group articles by cluster
    clusters = defaultdict(list)
    for article in articles:
        cluster_id = article.get('cluster', -1)
        
        # Skip noise points if they exist
        if cluster_id == -1:
            continue
        
        # Extract content based on available fields
        content = ""
        if "text" in article:
            content = article["text"]
        elif "content" in article:
            content = article["content"]
        elif "body_content" in article:
            content = article["body_content"]
        
        # Skip empty content
        if not content.strip():
            continue
            
        # Add title and content to the cluster
        title = article.get('title', 'No Title')
        
        clusters[cluster_id].append({
            'title': title,
            'content': content,
            'date': article.get('date', article.get('published_date', 'Unknown'))
        })
    
    # Process each cluster
    print(f"Processing TF-IDF for {len(clusters)} clusters")
    
    cluster_summaries = []
    
    for cluster_id, articles in clusters.items():
        print(f"Processing cluster {cluster_id} with {len(articles)} articles")
        
        # Get all content from the cluster
        texts = [clean_text(article['content']) for article in articles]
        titles = [article['title'] for article in articles]
        
        # Skip clusters with empty content
        if not any(text.strip() for text in texts):
            print(f"  Skipping cluster {cluster_id} - no valid content")
            continue
            
        try:
            # Configure TF-IDF vectorizer with more lenient parameters for small clusters
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=min(min_df, max(1, len(texts)//10)),  # Adjust min_df based on cluster size
                stop_words='english',
                ngram_range=(1, 2)  # Include bigrams for better topic representation
            )
            
            # Fit and transform the texts
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            if len(feature_names) == 0:
                print(f"  Warning: No features extracted for cluster {cluster_id}")
                continue
                
            # Calculate average TF-IDF score for each term across the cluster
            avg_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Create a sorted list of terms and scores
            tfidf_scores = [(feature_names[i], avg_tfidf[i]) for i in range(len(feature_names))]
            tfidf_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Calculate top terms for cluster summary
            top_terms = [term for term, score in tfidf_scores[:50]]
            cluster_summaries.append({
                'cluster_id': cluster_id,
                'size': len(articles),
                'top_terms': ', '.join(top_terms)
            })
            
            # Save to CSV
            csv_file = os.path.join("output_english/tfidf_clusters", f"cluster_{cluster_id}_tfidf.csv")
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Term', 'TF-IDF Score'])
                
                for term, score in tfidf_scores:
                    writer.writerow([term, score])
            
            # Save cluster articles details
            details_file = os.path.join("output_english/tfidf_clusters", f"cluster_{cluster_id}_articles.csv")
            with open(details_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Title', 'Date', 'Top Terms'])
                
                # For each article, calculate its individual top terms
                for idx, (article, text) in enumerate(zip(articles, texts)):
                    if not text.strip():
                        continue
                        
                    # Get the article's TF-IDF vector
                    article_vector = tfidf_matrix[idx].toarray()[0]
                    
                    # Get top terms for this article
                    article_terms = [(feature_names[i], article_vector[i]) 
                                    for i in range(len(feature_names)) if article_vector[i] > 0]
                    article_terms.sort(key=lambda x: x[1], reverse=True)
                    top_article_terms = ', '.join([term for term, _ in article_terms[:3]])
                    
                    writer.writerow([article['title'], article['date'], top_article_terms])
            
            print(f"  Saved TF-IDF for cluster {cluster_id} to {csv_file}")
            print(f"  Saved article details to {details_file}")
            
        except Exception as e:
            print(f"  Error processing cluster {cluster_id}: {e}")
    
    # Save cluster summaries
    summaries_file = os.path.join("output_english", "cluster_summaries.csv")
    pd.DataFrame(cluster_summaries).to_csv(summaries_file, index=False)
    print(f"Saved cluster summaries to {summaries_file}")

if __name__ == "__main__":
    # Get input file paths
    json_file = 'output_english/articles_with_clusters.json'
    
    csv_file = None
    if csv_file and not os.path.exists(csv_file):
        print(f"Warning: CSV file {csv_file} not found, proceeding with only JSON data")
        csv_file = None
    
    # Get TF-IDF parameters
    max_features = int(input("Enter maximum features per cluster (default: 100): ") or 100)
    min_df = int(input("Enter minimum document frequency (default: 2): ") or 2)
    
    # Process TF-IDF for clusters
    process_tfidf_for_clusters(json_file, csv_file, max_features, min_df)
    
    print("\nProcessing complete!")
    print("TF-IDF results saved to output_english/tfidf_clusters/ directory")