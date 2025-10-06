#!/usr/bin/env python3
import json
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import csv
import re
import jieba
import logging

# Configure jieba logging to make it less verbose
jieba.setLogLevel(logging.INFO)

def clean_text(text):
    """
    Clean text to remove URLs, special characters, and extra whitespace
    Optimized for Chinese text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove special characters but keep Chinese characters and spaces
    text = re.sub(r'[^\u4e00-\u9fff\s\w]', ' ', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def segment_chinese_text(text):
    """
    Segment Chinese text into words using jieba
    """
    if not isinstance(text, str):
        return ""
    
    # Clean the text first
    cleaned_text = clean_text(text)
    
    # Use jieba to segment Chinese text
    segmented_words = jieba.cut(cleaned_text)
    
    # Join with spaces to create format expected by TfidfVectorizer
    return " ".join(segmented_words)

def process_tfidf_for_clusters(input_file, csv_clusters_file=None, max_features=100, min_df=2):
    """
    Generate TF-IDF for each cluster in the input file
    Optimized for Chinese text
    
    Args:
        input_file: Path to JSON file with articles and cluster assignments
        csv_clusters_file: Optional path to CSV file with cluster assignments
        max_features: Maximum number of features to extract per cluster
        min_df: Minimum document frequency (to filter out rare terms)
    """
    print(f"Loading articles from {input_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs("output_chinese/tfidf_clusters", exist_ok=True)
    
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
        
        # Get all content from the cluster and segment using jieba
        texts = [segment_chinese_text(article['content']) for article in articles]
        titles = [article['title'] for article in articles]
        
        # Skip clusters with empty content
        if not any(text.strip() for text in texts):
            print(f"  Skipping cluster {cluster_id} - no valid content")
            continue
            
        try:
            # Load Chinese stopwords
            stopwords = list(load_chinese_stopwords())

            # Configure TF-IDF vectorizer - no stop_words parameter as we'll handle Chinese stop words differently
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=min(min_df, max(1, len(texts)//10)),  # Adjust min_df based on cluster size
                analyzer='word',
                token_pattern=r"(?u)\b\w+\b",  # This pattern works for space-separated text after jieba segmentation
                ngram_range=(1, 2) , # Include bigrams for better topic representation
                stop_words=stopwords 

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
            csv_file = os.path.join("output_chinese/tfidf_clusters", f"cluster_{cluster_id}_tfidf.csv")
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Term', 'TF-IDF Score'])
                
                for term, score in tfidf_scores:
                    writer.writerow([term, score])
            
            # Save cluster articles details
            details_file = os.path.join("output_chinese/tfidf_clusters", f"cluster_{cluster_id}_articles.csv")
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
    summaries_file = os.path.join("output_chinese", "cluster_summaries.csv")
    pd.DataFrame(cluster_summaries).to_csv(summaries_file, index=False)
    print(f"Saved cluster summaries to {summaries_file}")

def process_from_csv_directly(csv_file, max_features=100, min_df=2):
    """
    Process TF-IDF directly from a CSV file containing titles, dates, and cluster IDs
    This is useful when you don't have the full JSON data but just have article metadata
    
    Args:
        csv_file: Path to CSV file with article titles, dates, and cluster assignments
        max_features: Maximum number of features to extract per cluster
        min_df: Minimum document frequency (to filter out rare terms)
    """
    print(f"Loading articles from CSV: {csv_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs("output_chinese/tfidf_clusters", exist_ok=True)
    
    # Load the CSV data
    df = pd.read_csv(csv_file)
    
    # Make sure required columns exist
    required_cols = ['title', 'cluster']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain at least {required_cols} columns")
        return
    
    # Handle date column if it exists
    date_col = 'date' if 'date' in df.columns else None
    
    # Drop rows with cluster -1 (noise)
    df = df[df['cluster'] != -1]
    
    # Count articles per cluster
    cluster_counts = df['cluster'].value_counts().to_dict()
    
    print(f"Found {len(df)} articles across {len(cluster_counts)} clusters")
    
    # Since we only have titles, we'll use titles as content for TF-IDF
    # Group by cluster
    cluster_groups = df.groupby('cluster')
    
    cluster_summaries = []
    
    for cluster_id, group in cluster_groups:
        print(f"Processing cluster {cluster_id} with {len(group)} articles")
        
        # Get titles and segment using jieba
        titles = group['title'].tolist()
        segmented_titles = [segment_chinese_text(title) for title in titles]
        
        # Skip clusters with empty content
        if not any(title.strip() for title in segmented_titles):
            print(f"  Skipping cluster {cluster_id} - no valid titles")
            continue
            
        try:
            # Configure TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=min(min_df, max(1, len(segmented_titles)//10)),
                analyzer='word',
                token_pattern=r"(?u)\b\w+\b",
                ngram_range=(1, 2)
            )
            
            # Fit and transform the texts
            tfidf_matrix = vectorizer.fit_transform(segmented_titles)
            
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
            top_terms = [term for term, score in tfidf_scores[:5]]
            cluster_summaries.append({
                'cluster_id': cluster_id,
                'size': len(group),
                'top_terms': ', '.join(top_terms)
            })
            
            # Save to CSV
            csv_file = os.path.join("output_chinese/tfidf_clusters", f"cluster_{cluster_id}_tfidf.csv")
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Term', 'TF-IDF Score'])
                
                for term, score in tfidf_scores:
                    writer.writerow([term, score])
            
            # Save cluster articles details
            details_file = os.path.join("output_chinese/tfidf_clusters", f"cluster_{cluster_id}_articles.csv")
            with open(details_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                headers = ['Title']
                if date_col:
                    headers.append('Date')
                headers.append('Top Terms')
                
                writer.writerow(headers)
                
                # For each article, calculate its individual top terms
                for idx, title in enumerate(titles):
                    if not segmented_titles[idx].strip():
                        continue
                        
                    # Get the article's TF-IDF vector
                    article_vector = tfidf_matrix[idx].toarray()[0]
                    
                    # Get top terms for this article
                    article_terms = [(feature_names[i], article_vector[i]) 
                                    for i in range(len(feature_names)) if article_vector[i] > 0]
                    article_terms.sort(key=lambda x: x[1], reverse=True)
                    top_article_terms = ', '.join([term for term, _ in article_terms[:3]])
                    
                    row = [title]
                    if date_col:
                        row.append(group[date_col].iloc[idx])
                    row.append(top_article_terms)
                    
                    writer.writerow(row)
            
            print(f"  Saved TF-IDF for cluster {cluster_id} to {csv_file}")
            print(f"  Saved article details to {details_file}")
            
        except Exception as e:
            print(f"  Error processing cluster {cluster_id}: {e}")
    
    # Save cluster summaries
    summaries_file = os.path.join("output_chinese", "cluster_summaries.csv")
    pd.DataFrame(cluster_summaries).to_csv(summaries_file, index=False)
    print(f"Saved cluster summaries to {summaries_file}")

def load_chinese_stopwords():
    """
    Load Chinese stopwords from file or provide a default set
    """
    # Default minimal set of Chinese stopwords
    default_stopwords = {
        '的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
        '或', '一个', '没有', '我们', '你们', '他们', '它们', '这个',
        '那个', '这些', '那些', '不', '在', '有', '我', '你', '他','的'
    }
    #https://github.com/stopwords-iso/stopwords-zh/blob/master/stopwords-zh.txt
    # Try to load from file if it exists
    stopwords_file = 'topic_identification/chinese_stopwords.txt'
    if os.path.exists(stopwords_file):
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            stopwords = {line.strip() for line in f if line.strip()}
        return stopwords
    else:
        print(f"Stopwords file {stopwords_file} not found, using default stopwords")
        return default_stopwords

def add_user_dict_words(user_dict_file):
    """
    Add user dictionary words to jieba
    """
    if os.path.exists(user_dict_file):
        jieba.load_userdict(user_dict_file)
        print(f"Loaded user dictionary from {user_dict_file}")
    else:
        print(f"User dictionary file {user_dict_file} not found")

if __name__ == "__main__":
    print("Chinese TF-IDF Processor")
    print("========================")
    
    # Initialize jieba with user dictionary if available
    user_dict_file = 'chinese_user_dict.txt'
    add_user_dict_words(user_dict_file)
    
    # Select processing mode
    print("\nProcessing options:")
    print("1. Process from JSON file (articles_with_clusters.json)")
    print("2. Process directly from CSV file with titles and clusters")
    
    mode = input("Select option (1 or 2, default: 1): ") or "1"
    
    # Get TF-IDF parameters
    max_features = int(input("\nEnter maximum features per cluster (default: 100): ") or 100)
    min_df = int(input("Enter minimum document frequency (default: 2): ") or 2)
    
    if mode == "1":
        # Get input file paths
        json_file = input("\nEnter JSON file path (default: output_chinese/articles_with_clusters.json): ") or 'output_chinese/articles_with_clusters.json'
        
        csv_file = input("Enter optional CSV file with cluster assignments (leave empty to skip): ")
        if csv_file and not os.path.exists(csv_file):
            print(f"Warning: CSV file {csv_file} not found, proceeding with only JSON data")
            csv_file = None
        
        # Process TF-IDF for clusters
        process_tfidf_for_clusters(json_file, csv_file, max_features, min_df)
        
    else:  # mode == "2"
        # Get CSV file path
        csv_file = input("\nEnter CSV file path with titles and clusters: ")
        if not csv_file:
            print("Error: CSV file path is required")
            exit(1)
        
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} not found")
            exit(1)
            
        # Process TF-IDF directly from CSV
        process_from_csv_directly(csv_file, max_features, min_df)
    
    print("\nProcessing complete!")
    print("TF-IDF results saved to output_chinese/tfidf_clusters/ directory")

# todo
    # merge the translated prompt andrun this