import json
from collections import defaultdict


FOLDER = 'output_chinese'

# Load the JSON data from file
with open(f'{FOLDER}/articles_with_clusters.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Group data by cluster
cluster_dict = defaultdict(list)
for item in data:
    cluster_id = item.get("cluster")
    if cluster_id is not None:
        cluster_dict[cluster_id].append({
            "title": item.get("title", ""),
            "content": item.get("content", "")
        })

# Write each cluster's data to a separate text file
for cluster_id, items in cluster_dict.items():
    filename = f"{FOLDER}/cluster_{cluster_id}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(f"Title: {item['title']}\n")
            f.write(f"Content: {item['content']}\n")
            f.write("\n" + "-"*80 + "\n\n")

print("Filtered output files created for each cluster.")
