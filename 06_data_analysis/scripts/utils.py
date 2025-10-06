# general utilities for 06_data_analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, chi2_contingency

#################
# preprocessing
################


def create_wide_format_entropy(df):
    """
    Computes stance–distribution percentages and entropy per topic cluster, with both language-specific and language-collapsed views.
    """

    def process_grouped_entropy(df_grouped):
        # count stances per group
        stance_counts = (
            df_grouped.groupby(['cluster_id', 'topic_combined', 'language', 'topic_text','stance'])
            .size()
            .reset_index(name='count')
        )

        # pivot to wide format
        df_wide = stance_counts.pivot(index=[ 'cluster_id' ,'topic_combined', 'topic_text','language'], columns='stance', values='count').fillna(0)

        # ensure all stance bins exist
        for col in ['1', '2', '3', '4', '5', 'refusal']:
            if col not in df_wide.columns:
                df_wide[col] = 0

        df_wide = df_wide[['1', '2', '3', '4', '5', 'refusal']]

        # convert to percentages
        df_percent = df_wide.div(df_wide.sum(axis=1), axis=0) * 100

        # entropy
        df_percent['entropy'] = df_percent.apply(lambda row: entropy(row / row.sum(), base=2), axis=1)

        # collapsed bins
        df_percent['1+2'] = df_percent['1'] + df_percent['2']
        df_percent['4+5'] = df_percent['4'] + df_percent['5']

        # collapsed entropy
        def calc_collapsed_entropy(row):
            vals = [row['1+2'], row['3'], row['4+5']]
            probs = np.array(vals) / np.sum(vals)
            return entropy(probs, base=2)

        df_percent['entropy_collapsed'] = df_percent.apply(calc_collapsed_entropy, axis=1)
        df_percent['entropy_collapsed_n'] = df_percent['entropy_collapsed'] / np.log2(3)

        return df_percent.reset_index()

    # by language
    by_lang = process_grouped_entropy(df)

    # by overall (language collapsed)
    df_all = df.copy()
    df_all['language'] = 'all'
    overall = process_grouped_entropy(df_all)

    # combine
    final_df = pd.concat([by_lang, overall], ignore_index=True)
    final_df['media_source'] = final_df['cluster_id'].apply(lambda x: 'china' if 'c' in x else 'U.S')
    final_df['framing'] = final_df['topic_combined'].apply(lambda x: x.split('_')[-1])

    return final_df

##############
# for analysis
##############
def calculate_jsd(dist1, dist2):
    """
    Calculate Jensen-Shannon Divergence between two probability distributions.
    """
    from scipy.stats import entropy
    import numpy as np
    p = np.array(dist1)
    q = np.array(dist2)
    m = (p + q) / 2
    jsd = 0.5 * (entropy(p, m) + entropy(q, m))
    return jsd

def cramers_v(confusion_matrix: np.ndarray) -> float:
    """Calculate Cramer's V from a confusion matrix (bias-corrected)."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = np.asarray(confusion_matrix).sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

############
# latex
###########
