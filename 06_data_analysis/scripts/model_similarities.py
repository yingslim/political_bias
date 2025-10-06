
# local functions for this run_model_similarities.ipynb
import pandas as pd
import numpy as np
import seaborn as sns
from itertools import combinations
import matplotlib.pyplot as plt
from scripts.utils import calculate_jsd
from pathlib import Path


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
RESPONSE_COLS = ['1', '2', '3', '4', '5', 'refusal']
WESTERN_MODELS = ['Llama-3.3-70b-instruct', 'Gpt-4o-mini']
CHINESE_MODELS = ['Deepseek-chat-v3-0324', 'Qwen3-235b-a22b']
MODEL_NAME_MAPPING = {
    'Gpt-4o-mini': 'GPT-4o-mini',
    'Deepseek-chat-v3-0324': 'Deepseek-V3',
    'Qwen3-235b-a22b': 'Qwen-3',
    'Llama-3.3-70b-instruct': 'Llama-3.3',
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _filter_dataframe(df, topic_filter=None, language_filter=None, media_filter=None):
    """Apply filters to a dataframe."""
    filtered_df = df.copy()
    
    if topic_filter:
        filtered_df = filtered_df[filtered_df['topic_combined'].str.contains(topic_filter)]
    if language_filter:
        filtered_df = filtered_df[filtered_df['language'] == language_filter]
    if media_filter:
        filtered_df = filtered_df[filtered_df['cluster_id'].str.contains(media_filter)]
    
    return filtered_df


def _get_topic_distribution(model_df, topic_id, language_filter=None, media_filter=None):
    """
    Extract response distribution for a specific topic from a model's dataframe.
    
    Returns:
        np.array or None: Distribution vector, or None if no data found
    """
    subset = model_df[model_df['topic_combined'] == topic_id]
    
    if language_filter and language_filter != 'all':
        subset = subset[subset['language'] == language_filter]
    if media_filter:
        subset = subset[subset['cluster_id'].str.contains(media_filter)]
    
    if subset.empty:
        return None
    
    return subset[RESPONSE_COLS].iloc[0].values


def _get_averaged_distribution(model_list, topic_table_dict, topic_id, 
                               language_filter=None, media_filter=None):
    """
    Calculate average distribution across multiple models for a given topic.
    
    Returns:
        np.array or None: Averaged distribution, or None if no data available
    """
    distributions = []
    
    for model_name in model_list:
        if model_name in topic_table_dict:
            dist = _get_topic_distribution(
                topic_table_dict[model_name], 
                topic_id, 
                language_filter, 
                media_filter
            )
            if dist is not None:
                distributions.append(dist)
    
    if not distributions:
        return None
    
    return np.mean(distributions, axis=0)


def rename_and_reorder(df: pd.DataFrame,
                       name_map: dict,
                       order: list) -> pd.DataFrame:
    """Rename model rows/cols and reorder to desired order (dropping missing)."""
    df2 = df.rename(index=name_map, columns=name_map)
    present = [m for m in order if m in df2.index and m in df2.columns]
    return df2.reindex(index=present, columns=present)


def media_token_for_filename(m):
    return "all" if m is None else str(m)


def plot_similarity_matrix(mat: pd.DataFrame,
                           outpath: Path,
                           annot_fmt: str = ".3f"):
    """Plot similarity matrix heatmap and save to file."""
    plt.figure(figsize=(5, 5))
    sns.heatmap(
        mat, annot=True, fmt=annot_fmt,
        cmap="Blues", square=True,
        annot_kws={"fontsize": 8},
        cbar_kws={"shrink": 0.72}
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=900)
    plt.show()


# -----------------------------------------------------------------------------
# Main functions
#   - Per-topic similarity analysis
#   - Aggregate model similarities
#   - Divergence insights
# -----------------------------------------------------------------------------
def analyze_per_topic_similarities(topic_table_dict, 
                                   filter_models=None,
                                   filter_topics=None, 
                                   filter_language=None, 
                                   filter_media=None,
                                   group_by_origin=False):
    """
    Analyze model similarities for each individual topic.
    
    Calculates Jensen-Shannon Divergence (JSD) between models for each topic.
    Can compare individual models or group by origin (Western vs Chinese).
    
    Parameters:
        topic_table_dict (dict): Model name -> DataFrame mapping
        filter_models (list, optional): Specific models to include
        filter_topics (str, optional): String pattern to filter topics
        filter_language (str, optional): Language to filter by
        filter_media (str, optional): Media type to filter by
        group_by_origin (bool): If True, compare Western vs Chinese models
        
    Returns:
        pd.DataFrame: Per-topic similarity scores with columns:
            - language, media, framing, topic_combined, topic_text
            - If group_by_origin=False: avg_jsd, max_jsd, min_jsd, std_jsd
            - If group_by_origin=True: jsd, comparison
    """
    # Filter models if specified
    if filter_models is not None:
        topic_table_dict = {m: topic_table_dict[m] for m in filter_models}
    
    model_names = list(topic_table_dict.keys())
    
    # Get reference DataFrame and apply filters
    ref_df = topic_table_dict[model_names[0]]
    ref_df = _filter_dataframe(ref_df, filter_topics, filter_language, filter_media)
    
    results = []
    
    # Process each topic
    for _, row in ref_df.iterrows():
        topic_id = row['topic_combined']
        
        if group_by_origin:
            # Compare Western vs Chinese model groups
            western_dist = _get_averaged_distribution(
                WESTERN_MODELS, topic_table_dict, topic_id, 
                filter_language, filter_media
            )
            chinese_dist = _get_averaged_distribution(
                CHINESE_MODELS, topic_table_dict, topic_id,
                filter_language, filter_media
            )
            
            if western_dist is None or chinese_dist is None:
                continue
            
            jsd = calculate_jsd(western_dist, chinese_dist)
            
            results.append({
                'language': row['language'],
                'media': row['media_source'],
                'framing': row['framing'],
                'topic_combined': topic_id,
                'topic_text': row['topic_text'],
                'jsd': jsd,
                'comparison': 'Western vs Chinese'
            })
        else:
            # Compare individual models pairwise
            distributions = []
            for model_name in model_names:
                dist = _get_topic_distribution(
                    topic_table_dict[model_name], topic_id,
                    filter_language, filter_media
                )
                if dist is not None:
                    distributions.append(dist)
            
            if len(distributions) < 2:
                continue
            
            # Calculate JSD for all pairs
            jsds = [
                calculate_jsd(distributions[i], distributions[j])
                for i, j in combinations(range(len(distributions)), 2)
            ]
            
            results.append({
                'language': row['language'],
                'media': row['media_source'],
                'framing': row['framing'],
                'topic_combined': topic_id,
                'topic_text': row['topic_text'],
                'avg_jsd': np.mean(jsds),
                'max_jsd': np.max(jsds),
                'min_jsd': np.min(jsds),
                'std_jsd': np.std(jsds)
            })
    
    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results)
    sort_col = 'jsd' if group_by_origin else 'avg_jsd'
    results_df = results_df.sort_values(sort_col)
    
    return results_df


def calculate_aggregate_model_similarities(topic_table_dict,
                                          filter_topics=None,
                                          filter_language=None, 
                                          filter_media=None):
    """
    Calculate pairwise similarities between models aggregated across all topics.
    
    Computes average JSD between each pair of models across all topics.
    
    Parameters:
        topic_table_dict (dict): Model name -> DataFrame mapping
        filter_topics (str, optional): String pattern to filter topics
        filter_language (str, optional): Language to filter by
        filter_media (str, optional): Media type to filter by
        
    Returns:
        tuple: (pairwise_df, similarity_matrix)
            - pairwise_df: DataFrame with columns [model1, model2, avg_jsd, 
              median_jsd, std_jsd, min_jsd, max_jsd]
            - similarity_matrix: Symmetric DataFrame of average JSDs
    """
    model_names = list(topic_table_dict.keys())
    
    # Initialize storage for pairwise JSDs
    pairwise_jsds = {pair: [] for pair in combinations(model_names, 2)}
    
    # Get reference DataFrame and apply filters
    ref_df = topic_table_dict[model_names[0]]
    ref_df = _filter_dataframe(ref_df, filter_topics, filter_language, filter_media)
    
    # Calculate JSD for each model pair across all topics
    for _, row in ref_df.iterrows():
        topic_id = row['topic_combined']
        
        for model1, model2 in combinations(model_names, 2):
            dist1 = _get_topic_distribution(
                topic_table_dict[model1], topic_id, 
                filter_language, filter_media
            )
            dist2 = _get_topic_distribution(
                topic_table_dict[model2], topic_id,
                filter_language, filter_media
            )
            
            if dist1 is None or dist2 is None:
                continue
            
            jsd = calculate_jsd(dist1, dist2)
            pairwise_jsds[(model1, model2)].append(jsd)
    
    # Calculate summary statistics
    results = []
    for (model1, model2), jsds in pairwise_jsds.items():
        if not jsds:  # Skip if no valid comparisons
            continue
            
        results.append({
            'model1': model1,
            'model2': model2,
            'avg_jsd': np.mean(jsds),
            'median_jsd': np.median(jsds),
            'std_jsd': np.std(jsds),
            'min_jsd': np.min(jsds),
            'max_jsd': np.max(jsds),
            'n_topics': len(jsds)
        })
    
    pairwise_df = pd.DataFrame(results).sort_values('avg_jsd')
    
    # Create symmetric similarity matrix
    similarity_matrix = pd.DataFrame(
        0.0, 
        index=model_names, 
        columns=model_names
    )
    
    for _, row in pairwise_df.iterrows():
        similarity_matrix.loc[row['model1'], row['model2']] = row['avg_jsd']
        similarity_matrix.loc[row['model2'], row['model1']] = row['avg_jsd']
    
    return pairwise_df, similarity_matrix


def generate_highest_divergence_tables(neutral_framed, top_n=5):
    """
    to get the top 5 divergent issues between English and Mandarin for each model
    """
    expected_cols = ['1','2','3','4','5','refusal']

    def escape_latex(text):
        replacements = {
            '_': r'\_',
            '/': r'\/',
            '%': r'\%',
            '&': r'\&',
            '#': r'\#',
            '{': r'\{',
            '}': r'\}',
            '$': r'\$',
            '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}',
            '\\': r'\textbackslash{}',
        }
        for char, repl in replacements.items():
            text = text.replace(char, repl)
        return text
    
    def format_latex_row(issue_en, issue_zh, row, is_gray):
        en_vals = row[[f"{c}_en" for c in expected_cols]].to_numpy(dtype=float)
        zh_vals = row[[f"{c}_zh" for c in expected_cols]].to_numpy(dtype=float)

        # normalize
        en_vals = en_vals / en_vals.sum() if en_vals.sum() > 0 else en_vals
        zh_vals = zh_vals / zh_vals.sum() if zh_vals.sum() > 0 else zh_vals

        # build bars
        bar_en = r"\barrule{" + "}{".join([f"{v:.4f}" for v in en_vals]) + "}"
        bar_zh = r"\barrule{" + "}{".join([f"{v:.4f}" for v in zh_vals]) + "}"

        rowcolor = r"\rowcolor[HTML]{EFEFEF} " if is_gray else ""
        return f"{rowcolor}{escape_latex(issue_en)} & {bar_en} & {escape_latex(issue_zh)} & {bar_zh} \\\\"

    for model in neutral_framed['model'].unique():
        df_model = neutral_framed[neutral_framed['model'] == model]

        # split english vs mandarin
        df_en = df_model[df_model['language'] == 'english']
        df_zh = df_model[df_model['language'] == 'mandarin']

        # stance distributions grouped by cluster_id
        stance_en = df_en.groupby('cluster_id')['stance'].value_counts(normalize=True).unstack(fill_value=0)
        stance_zh = df_zh.groupby('cluster_id')['stance'].value_counts(normalize=True).unstack(fill_value=0)
        for c in expected_cols:
            if c not in stance_en.columns:
                stance_en[c] = 0
            if c not in stance_zh.columns:
                stance_zh[c] = 0
        stance_en = stance_en[expected_cols].reset_index()
        stance_zh = stance_zh[expected_cols].reset_index()

        # issue text
        issue_en_df = df_en[['cluster_id','issue']].drop_duplicates('cluster_id').rename(columns={'issue':'issue_en'})
        issue_zh_df = df_zh[['cluster_id','issue']].drop_duplicates('cluster_id').rename(columns={'issue':'issue_zh'})

        # merge together
        merged = stance_en.merge(issue_en_df, on='cluster_id', how='left')
        merged = merged.merge(stance_zh, on='cluster_id', how='left', suffixes=('_en','_zh'))
        merged = merged.merge(issue_zh_df, on='cluster_id', how='left')

        # compute divergence
        divergences = []
        for _, row in merged.iterrows():
            en_vals = row[[f"{c}_en" for c in expected_cols]].to_numpy(dtype=float)
            zh_vals = row[[f"{c}_zh" for c in expected_cols]].to_numpy(dtype=float)
            jsd_val = calculate_jsd(en_vals, zh_vals)
            divergences.append(jsd_val)
        merged['divergence'] = divergences
        merged = merged.sort_values('divergence', ascending=False).head(top_n)

        latex_rows = []
        for i, (_, row) in enumerate(merged.iterrows()):
            # print(row)
            latex_rows.append(format_latex_row(row['issue_en'], row['issue_zh'], row, is_gray=(i % 2 == 0)))

        latex_body = "\n".join(latex_rows)
        model_display = escape_latex(model)
        caption = f"\\textbf{{Top {top_n} issues with largest stance divergence between English and Mandarin for {model_display}.}}"

        latex_table = f"""
        % ========== MODEL: {model_display} ==========
        \\begin{{table}}[H]
        \\small
        \\centering
        \\renewcommand{{\\arraystretch}}{{1.2}}
        \\resizebox{{\\linewidth}}{{!}}{{%
        \\begin{{tabular}}{{p{{6cm}}l p{{6cm}}l}}
        \\toprule
        \\textbf{{Issue (English)}} & \\textbf{{English Stance}} & \\textbf{{Issue (Mandarin)}} & \\textbf{{Mandarin Stance}} \\\\
        \\midrule
        {latex_body}
        \\bottomrule
        \\end{{tabular}}
        }}
        \\caption{{{caption}}}
        \\label{{tab:stance-divergence-{model.replace('/','-').replace('_','-')}}}
        \\end{{table}}
        """
        print(latex_table)


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------
def plot_jsd_histogram(results_by_combo, key, jsd_col):
    """
    Plot JSD histogram of issues for a given key (language, media, framing)
    """
    res = results_by_combo[key].sort_values(jsd_col, ascending=False) #.head(5)
    print(len(res))
    print(f"Plotting histogram for {key}")
    # Create the figure
    plt.figure(figsize=(5, 2))

    # Create the histogram with seaborn
    sns.histplot(
        data=res,
        x=jsd_col,
        bins=15,
        binrange=(0, 0.6),
        edgecolor='#1865ac',
        color='#1865ac',  # A pleasant blue color
        alpha=1
    )

    # Customize the axes
    plt.ylim(0, 30)          # lock y-axis to 0..30
    plt.xlim(0, 0.6)
    plt.xlabel('')
    plt.ylabel('No. of Issues', fontsize=12)

    # Remove spines
    sns.despine()

    # Adjust layout
    plt.tight_layout()

    plt.savefig(f'figs/jsd_histogram_{key[0]}_{key[1]}_{key[2]}.png', dpi=900)

    plt.show()


def make_horizontal_bar_chart(model, row, show_latex=True, show_plot=True):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 0.75))

    ax.barh(row.topic_text, row["1"], color="#38761dff", label="only pro")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.barh(row.topic_text, row["2"], left=row["1"], color="#93c47dff", label="mostly pro")
    ax.barh(row.topic_text, row["3"], left=row["1"]+row["2"], color="#ffd966ff", label="ambivalent")
    ax.barh(row.topic_text, row["4"], left=row["1"]+row["2"]+row["3"], color="#e06666ff", label="mostly con")
    ax.barh(row.topic_text, row["5"], left=row["1"]+row["2"]+row["3"]+row["4"], color="#cc0000ff", label="only con")
    ax.barh(row.topic_text, row["refusal"], left=row["1"]+row["2"]+row["3"]+row["4"]+row["5"], color="#b7b7b7ff", label="refusal")
    ax.set_yticklabels([])
    
    # set title 
    ax.set_title(row.topic_text, fontweight="bold")

    if show_plot:
        plt.show()

    if show_latex:
        topic = row.topic_text
        # Escape special characters for LaTeX
        topic_escaped = topic.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

        # Print the topic as a separate bold row across columns
        print(r"\multicolumn{2}{l}{\textbf{" + topic_escaped + r"}} \\")
        print(r"\midrule")  # optional line after topic header

        # Then print the model row
        print(f"{model} & \\barrule{{{row['1']/100:.4f}}}{{{row['2']/100:.4f}}}{{{row['3']/100:.4f}}}{{{row['4']/100:.4f}}}{{{row['5']/100:.4f}}}{{{row['refusal']/100:.4f}}} \\\\")
