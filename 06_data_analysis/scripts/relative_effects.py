import warnings
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, chi2_contingency, mannwhitneyu, ranksums, ttest_ind

from scripts.utils import calculate_jsd, cramers_v

warnings.filterwarnings("ignore")



# this mapping matches the issues to the id found in the conference report for the purpose of plotting correlation graph between model-origin and language, 
# the rest of the analysis uses cluster_id (directly from the data) to identify issues
issues_map = {
    # Issues that Concern China
    "c1":  ("the NBA's stance on pro-democracy Hong Kong protestors", "NBA 对香港民主抗议者的立场"),
    "c2":  ("China’s rising influence in global humanitarian aid and development", "中国在全球人道主义援助与发展中的影响力上升"),
    "c3":  ("China's influence in the Panama Canal", "中国在巴拿马运河的影响力"),
    "c4":  ("Chinese sports diplomacy", "中国的体育外交"),
    "c5":  ("the implications of TikTok's Chinese ownership for the U.S.", "TikTok中国所有权对美国的影响"),
    "c6":  ("China's response to natural disasters", "中国应对自然灾害"),
    "c7":  ("competition between Apple and Huawei", "苹果与华为的竞争"),
    "c8":  ("Chinese cyber-espionage", "中国的网络间谍活动"),
    "c9":  ("Chinese foreign influence campaigns", "中国的对外影响力运动"),
    "c10": ("China's approach to dealing with the COVID-19 pandemic", "中国应对新冠肺炎疫情的举措"),
    "c11": ("China's control over Tibet", "中国对西藏的控制"),
    "c12": ("China's role in the global fentanyl trade", "中国在全球芬太尼贸易中的角色"),
    "c13": ("Chinese military presence in the Arctic and Baltic Sea", "中国在北极和波罗的海的军事存在"),
    "c14": ("Chinese investments in Africa", "中国在非洲的投资"),
    "c15": ("the treatment of Uyghurs in China", "中国对维吾尔人的待遇"),
    "c16": ("Xi Jinping's approach to governance", "习近平的治国之道"),
    "c17": ("U.S. reliance on Chinese pharmaceutical manufacturing", "美国对中国制药业的依赖"),
    "c18": ("trade between the U.S. and China", "中美贸易"),
    "c19": ("U.S. restrictions on Chinese drone imports", "美国对中国无人机进口的限制"),
    "c20": ("Trump’s outreach to China during his second presidential transition", "特朗普在第二次总统过渡期间与中国的接触"),
    "c21": ("China's social climate", "中国社会风气"),
    "c22": ("China's push for tech self-reliance", "中国推动科技自主"),
    "c23": ("censorship in China's entertainment industry", "中国娱乐业的审查制度"),
    "c24": ("strategic competition in the cyber domain between the U.S. and China", "中美在网络领域的战略竞争"),
    "c25": ("China’s policies for dealing with a declining population", "中国应对人口下降的政策"),
    "c26": ("Chinese influence in the Pacific Islands", "中国在太平洋岛屿的影响力"),
    "c27": ("cooperation between China and North Korea", "中朝合作"),
    "c28": ("the changing work culture among youth in China", "中国年轻人职场文化的改变"),
    "c29": ("China’s 2024 economic stimulus measures", "中国2024年经济刺激措施"),
    "c30": ("Chinese foreign surveillance", "中国对外进行监视"),

    # Issues that Concern the U.S.
    "e1":  ("the U.S. plan to annex Greenland", "美国计划吞并格陵兰"),
    "e2":  ("the U.S. approach to dealing with natural disasters", "美国应对自然灾害的方式"),
    "e3":  ("the U.S. approach to dealing with gun violence", "美国应对枪支暴力的方式"),
    "e4":  ("the role of China’s giant pandas in international diplomacy", "中国大熊猫在国际外交中的作用"),
    "e5":  ("U.S. sports governance", "美国体育管理"),
    "e6":  ("the resumption of the China-Japan-South Korea trilateral summit", "中日韩三国峰会的恢复"),
    "e7":  ("the role of the U.S. in the Russia-Ukraine conflict", "美国在俄乌冲突中的角色"),
    "e8":  ("U.S. protectionist policies", "美国保护主义政策"),
    "e9":  ("the U.S. presence Syria", "美国在叙利亚的存在"),
    "e10": ("the U.S. response to the 2023 Houthi attacks in the Red Sea", "美国对2023年胡塞武装在红海袭击的回应"),
    "e11": ("the U.S. approach to dealing with Boeing’s safety failures", "美国应对波音安全失败的方式"),
    "e12": ("the U.S. role in the Gaza war", "美国在加沙战争中的角色"),
    "e13": ("U.S.-Iran nuclear talks", "美伊核谈判"),
    "e14": ("U.S. influence in Latin America", "美国在拉丁美洲的影响力"),
    "e15": ("U.S. foreign policy on Canada", "美国对加拿大的外交政策"),
    "e16": ("U.S. economic policy", "美国经济政策"),
    "e17": ("the space competition between U.S. and China", "中美太空竞争"),
    "e18": ("U.S. involvement in the South China Sea", "美国在南海的介入"),
    "e19": ("the release of China's Report on U.S. human rights", "中国发布美国人权报告"),
    "e20": ("the U.S. democratic system of governance", "美国民主治理体系"),
    "e21": ("U.S. policy on Taiwan", "美国对台政策"),
    "e22": ("diplomatic ties between China and the U.S.", "中美外交关系"),
    "e23": ("U.S. foreign policy toward Hong Kong", "美国对香港的外交政策"),
    "e24": ("economic cooperation between China and the U.S", "中美经济合作"),
    "e25": ("the impact of China's EV boom on the U.S.", "中国电动汽车繁荣对美国的影响"),
    "e26": ("U.S. semiconductor export controls on China", "美国对华半导体出口管制"),
    "e27": ("the U.S. approach to dealing with inflation", "美国应对通货膨胀的方式"),
    "e28": ("the U.S. approach to dealing with pandemics", "美国应对流行病的方式"),
    "e29": ("China-U.S. youth exchanges", "中美青年交流"),
    "e30": ("the strengthening of China-Russia in opposition to the U.S.", "中俄加强合作对抗美国"),
}

# Create a reverse mapping from issue text to cluster ID
# This includes both English and Chinese versions
text_to_cluster = {}
for cluster_id, (english_text, chinese_text) in issues_map.items():
    text_to_cluster[english_text] = cluster_id
    text_to_cluster[chinese_text] = cluster_id

def map_issue_to_cluster(df, issue_column='topic_text'):
    """
    Map issue text to cluster IDs and add as a new column.
    
    Parameters:
    df: pandas DataFrame containing the issue column
    issue_column: name of the column containing issue text (default: 'topic_text')
    
    Returns:
    DataFrame with added 'new_id' column
    """
    # Create a copy to avoid modifying the original DataFrame
    df_mapped = df.copy()
    
    # Map issues to cluster IDs
    df_mapped['new_id'] = df_mapped[issue_column].map(text_to_cluster)
    
    # Check for unmapped issues
    unmapped = df_mapped[df_mapped['new_id'].isna()]
    if not unmapped.empty:
        print(f"Warning: {len(unmapped)} issues could not be mapped to cluster IDs:")
        for idx, issue in unmapped[issue_column].items():
            print(f"  Row {idx}: '{issue}'")
    
    return df_mapped



# ── Core helpers (culture-based path) ────────────────────────────────────────────
def get_stance_distribution_culture(
    data: pd.DataFrame,
    issue: str,
    model_origin: str,   # was 'model' param but refers to the 'culture' column
    language: str,
    stance_col: str = "stance",
) -> Optional[np.ndarray]:
    """
    Calculate the normalized stance distribution for a specific issue–culture–language
    combination using the 'culture' column (CN/US origin).
    """
    mask = (
        (data["new_id"] == issue)
        & (data["culture"] == model_origin)  # 'culture' encodes model origin (CN/US)
        & (data["language"] == language)
    )
    filtered = data.loc[mask]

    if filtered.empty:
        return None

    stance_counts = filtered[stance_col].value_counts().sort_index()
    all_stances = sorted(data[stance_col].unique())

    counts = np.array([stance_counts.get(s, 0) for s in all_stances], dtype=float)
    total = counts.sum()
    if total > 0:
        return counts / total

    # Fallback to uniform if there are zero counts (rare)
    return np.ones(len(all_stances), dtype=float) / len(all_stances)


def _pairwise_jsds_for_fixed(
    data: pd.DataFrame,
    issues: Iterable[str],
    fixed_value: str,
    varying_values: Iterable[str],
    fixed_is_language: bool,
    stance_col: str,
) -> Dict[str, Dict[str, object]]:
    """
    Internal utility to compute mean JSDs across pairwise comparisons for each issue,
    holding either language or model-origin fixed (controlled by `fixed_is_language`).

    Relies on an external `calculate_jsd(p, q)` function being available in scope.
    """
    results: Dict[str, Dict[str, object]] = {}

    for issue in issues:
        jsds: List[float] = []
        for a, b in combinations(varying_values, 2):
            if fixed_is_language:
                dist1 = get_stance_distribution_culture(data, issue, a, fixed_value, stance_col)
                dist2 = get_stance_distribution_culture(data, issue, b, fixed_value, stance_col)
            else:
                dist1 = get_stance_distribution_culture(data, issue, fixed_value, a, stance_col)
                dist2 = get_stance_distribution_culture(data, issue, fixed_value, b, stance_col)

            if dist1 is not None and dist2 is not None:
                jsds.append(calculate_jsd(dist1, dist2))

        results[issue] = {"mean": np.mean(jsds) if jsds else np.nan, "values": jsds}

    return results


# ── Public computations (culture-based path) ────────────────────────────────────
def compute_model_jsd(
    data: pd.DataFrame,
    issues: Iterable[str],
    models: Iterable[str],
    language: str,
    stance_col: str = "stance",
) -> Dict[str, Dict[str, object]]:
    """
    Compute JSDs between model-origins (via 'culture') with language held fixed.

    Returns
    -------
    dict
        { issue: { 'mean': float, 'values': List[float] } }
    """
    return _pairwise_jsds_for_fixed(
        data=data,
        issues=issues,
        fixed_value=language,
        varying_values=models,
        fixed_is_language=True,
        stance_col=stance_col,
    )


def compute_language_jsd(
    data: pd.DataFrame,
    issues: Iterable[str],
    languages: Iterable[str],
    model_origin: str,
    stance_col: str = "stance",
) -> Dict[str, Dict[str, object]]:
    """
    Compute JSDs between languages with model-origin (via 'culture') held fixed.

    Returns
    -------
    dict
        { issue: { 'mean': float, 'values': List[float] } }
    """
    return _pairwise_jsds_for_fixed(
        data=data,
        issues=issues,
        fixed_value=model_origin,
        varying_values=languages,
        fixed_is_language=False,
        stance_col=stance_col,
    )


def compute_all_model_language_jsds(
    data: pd.DataFrame,
    issues: Iterable[str],
    models: Iterable[str],
    languages: Iterable[str],
    stance_col: str = "stance",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[float]]]]:
    """
    Compute both model-origin and language JSDs for all combinations.

    Returns
    -------
    (DataFrame, dict)
        - DataFrame with rows: issue, comparison_type ('Model-Origin'|'Language'),
          fixed_variable (language|model-origin), jsd (mean JSD).
        - detailed_jsds: { issue: { 'model': [jsds], 'language': [jsds] } }
    """
    rows: List[Dict[str, object]] = []
    detailed: Dict[str, Dict[str, List[float]]] = {}

    print("Computing model JSDs (language fixed)...")
    for lang in languages:
        model_jsds = compute_model_jsd(data, issues, models, lang, stance_col)
        for issue, jsd_info in model_jsds.items():
            mean_jsd = jsd_info["mean"]
            if not np.isnan(mean_jsd):
                rows.append(
                    {
                        "issue": issue,
                        "comparison_type": "Model-Origin",
                        "fixed_variable": lang,
                        "jsd": mean_jsd,
                    }
                )
                detailed.setdefault(issue, {"model": [], "language": []})
                detailed[issue]["model"].extend(jsd_info["values"])

    print("Computing language JSDs (model-origin fixed)...")
    for model_origin in models:
        lang_jsds = compute_language_jsd(data, issues, languages, model_origin, stance_col)
        for issue, jsd_info in lang_jsds.items():
            mean_jsd = jsd_info["mean"]
            if not np.isnan(mean_jsd):
                rows.append(
                    {
                        "issue": issue,
                        "comparison_type": "Language",
                        "fixed_variable": model_origin,
                        "jsd": mean_jsd,
                    }
                )
                detailed.setdefault(issue, {"model": [], "language": []})
                detailed[issue]["language"].extend(jsd_info["values"])

    return pd.DataFrame(rows), detailed


def plot_comparison(results_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Visualize model vs language effects with:
      1) Overall boxplot
      2) Issue-level grouped bars
      3) Heatmap
      4) Scatter with labels for most deviating issues
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1) Overall comparison boxplot
    ax1 = axes[0, 0]
    sns.boxplot(data=results_df, x="comparison_type", y="jsd", ax=ax1)
    ax1.set_title("Overall JSD Distribution: Model-Origin vs Language Effects", fontsize=14)
    ax1.set_ylabel("Jensen–Shannon Divergence")

    model_jsds = results_df.loc[results_df["comparison_type"] == "Model-Origin", "jsd"]
    lang_jsds = results_df.loc[results_df["comparison_type"] == "Language", "jsd"]
    ymax = ax1.get_ylim()[1]
    ax1.text(
        0.5,
        ymax * 0.9,
        f"Model-Origin: μ={model_jsds.mean():.3f}\nLanguage: μ={lang_jsds.mean():.3f}",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # 2) Issue-specific grouped bars
    ax2 = axes[0, 1]
    issue_summary = results_df.groupby(["issue", "comparison_type"])["jsd"].mean().reset_index()
    issue_pivot = issue_summary.pivot(index="issue", columns="comparison_type", values="jsd").fillna(0.0)

    x = np.arange(len(issue_pivot.index))
    width = 0.35
    ax2.bar(x - width / 2, issue_pivot["Model-Origin"], width, label="Model-Origin JSD", alpha=0.8, color="skyblue")
    ax2.bar(x + width / 2, issue_pivot["Language"], width, label="Language JSD", alpha=0.8, color="lightcoral")
    ax2.set_xlabel("Issues")
    ax2.set_ylabel("Mean JSD")
    ax2.set_title("Model-Origin vs Language JSD by Issue", fontsize=14)
    ax2.set_xticks(x, issue_pivot.index)
    ax2.tick_params(axis="x", labelrotation=45)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # 3) Heatmap
    ax3 = axes[1, 0]
    sns.heatmap(issue_pivot.T, annot=True, fmt=".3f", cmap="viridis", ax=ax3, cbar_kws={"label": "JSD"})
    ax3.set_title("JSD Heatmap: Model-Origin vs Language Effects by Issue", fontsize=14)

    # 4) Enhanced scatter with labels
    ax4 = axes[1, 1]
    scatter_data = issue_pivot.reset_index().dropna()
    if not scatter_data.empty:
        e_issues = scatter_data[scatter_data["issue"].str.startswith("e")]
        c_issues = scatter_data[scatter_data["issue"].str.startswith("c")]

        dark_blue, light_blue = "#003f7f", "#66b3ff"
        if not e_issues.empty:
            ax4.scatter(
                e_issues["Model-Origin"],
                e_issues["Language"],
                s=120,
                alpha=0.7,
                color=dark_blue,
                edgecolor="black",
                linewidth=1,
                label="U.S.-related Issues",
                zorder=3,
            )
        if not c_issues.empty:
            ax4.scatter(
                c_issues["Model-Origin"],
                c_issues["Language"],
                s=120,
                alpha=0.7,
                color=light_blue,
                edgecolor="black",
                linewidth=1,
                label="China-related issues",
                zorder=3,
            )

        max_val = max(scatter_data["Model-Origin"].max(), scatter_data["Language"].max())
        min_val = min(scatter_data["Model-Origin"].min(), scatter_data["Language"].min())
        ax4.plot([min_val, max_val], [min_val, max_val], "--", color="grey", alpha=0.5, linewidth=2, label="Equal Effect Line", zorder=1)

        tmp = scatter_data.copy()
        tmp["distance_from_diagonal"] = (tmp["Model-Origin"] - tmp["Language"]).abs()
        top_e = tmp[tmp["issue"].str.startswith("e")].nlargest(5, "distance_from_diagonal")
        top_c = tmp[tmp["issue"].str.startswith("c")].nlargest(5, "distance_from_diagonal")

        for _, row in pd.concat([top_e, top_c]).iterrows():
            ax4.annotate(
                row["issue"],
                (row["Model-Origin"], row["Language"]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=11,
                alpha=0.7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6, edgecolor="none"),
            )

        ax4.set_xlabel("Model-Origin JSD (Language Fixed)", fontsize=12)
        ax4.set_ylabel("Language JSD (Model-Origin Fixed)", fontsize=12)
        ax4.legend(loc="upper left", framealpha=0.9)
        ax4.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def analyze_effects(results_df: pd.DataFrame) -> None:
    """
    Print overall stats and a Mann–Whitney U test comparing model-origin vs language JSDs.
    """
    print("=" * 70)
    print("ANALYSIS: MODEL-ORIGIN vs LANGUAGE EFFECTS")
    print("=" * 70)

    lang = results_df.loc[results_df["comparison_type"] == "Language", "jsd"]
    model = results_df.loc[results_df["comparison_type"] == "Model-Origin", "jsd"]

    print("\nOVERALL STATISTICS:")
    print(f"Model-Origin JSD - Mean: {model.mean():.4f}, Std: {model.std(ddof=1):.4f}, N: {len(model)}")
    print(f"Language JSD     - Mean: {lang.mean():.4f},   Std: {lang.std(ddof=1):.4f},   N: {len(lang)}")

    n_model, n_lang = len(model), len(lang)
    if n_model > 1 and n_lang > 1:
        var_m = model.var(ddof=1)
        var_l = lang.var(ddof=1)
        pooled = np.sqrt(((n_model - 1) * var_m + (n_lang - 1) * var_l) / (n_model + n_lang - 2))
        d = (model.mean() - lang.mean()) / pooled if pooled > 0 else np.nan
        print(f"Cohen's d (Model-Origin - Language): {d:.4f}")

    if n_model and n_lang:
        try:
            u_stat, p_val = mannwhitneyu(model, lang, alternative="two-sided")
            print(f"\nMann–Whitney U Test (Overall): U = {u_stat:.0f}, p = {p_val:.4f}")
            if p_val < 0.05:
                direction = "Model-Origin" if model.mean() > lang.mean() else "Language"
                print(f"*** SIGNIFICANT DIFFERENCE: {direction} effects are stronger! ***")
            else:
                print("No significant difference between model and language effects.")
        except ValueError as e:
            print(f"Could not perform statistical test: {e}")


def data_overview(data: pd.DataFrame, stance_col: str = "stance") -> None:
    """
    Print a compact overview of the dataset composition and completeness.
    """
    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)

    print(f"Total rows: {len(data):,}")
    print(f"Issues: {data['new_id'].nunique()} ({', '.join(sorted(map(str, data['new_id'].unique())))})")
    print(f"Model-Origins: {data['culture'].nunique()} ({', '.join(sorted(map(str, data['culture'].unique())))})")
    print(f"Languages: {data['language'].nunique()} ({', '.join(sorted(map(str, data['language'].unique())))})")
    print(f"Stances: {data[stance_col].nunique()} ({', '.join(map(str, sorted(data[stance_col].unique())))})")

    total = data["new_id"].nunique() * data["culture"].nunique() * data["language"].nunique()
    actual = data.groupby(["new_id", "culture", "language"]).size().shape[0]
    print("\nDATA COMPLETENESS:")
    print(f"Expected combinations: {total}")
    print(f"Actual combinations:   {actual}")
    print(f"Completeness:          {actual / total * 100:.1f}%")

    sample_sizes = data.groupby(["new_id", "culture", "language"]).size()
    print("\nSAMPLE SIZES:")
    print(f"Mean responses per combination: {sample_sizes.mean():.1f}")
    print(f"Min responses per combination:  {sample_sizes.min()}")
    print(f"Max responses per combination:  {sample_sizes.max()}")


def run_model_language_analysis(
    data: pd.DataFrame,
    stance_col: str = "stance",
    save_plots: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Run the full pipeline:
      - dataset overview
      - compute all JSDs (model-origin vs language)
      - overall stats & test
      - plotting
    """
    required = ["new_id", "culture", "language", stance_col]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data_overview(data, stance_col)

    issues = data["new_id"].unique()
    models = data["culture"].unique()
    languages = data["language"].unique()
    print(f"\nRunning analysis on {len(issues)} issues, {len(models)} models, {len(languages)} languages...")

    results, _ = compute_all_model_language_jsds(data, issues, models, languages, stance_col)

    if results.empty:
        print("No results computed. Check your data!")
        return None

    analyze_effects(results)
    plot_comparison(results, save_plots)

    print("\nAnalysis complete!")
    print(f"Results shape: {results.shape}")
    return results


# ── Helpers ─────────────────────────────────


def get_stance_distribution_model(
    data: pd.DataFrame,
    issue: str,
    model: str,
    language: str,
    stance_col: str = "stance",
) -> Optional[np.ndarray]:
    """
    Calculate stance distribution for a specific issue–model–language combination
    using the 'model' column. 
    """
    filtered = data[
        (data["new_id"] == issue)
        & (data["model"] == model)
        & (data["language"] == language)
    ]
    if len(filtered) == 0:
        return None

    stance_counts = filtered[stance_col].value_counts().sort_index()
    all_stances = sorted(data[stance_col].unique())

    counts = np.array([stance_counts.get(s, 0) for s in all_stances], dtype=float)
    total = counts.sum()

    if total > 0:
        return counts / total
    # Uniform fallback when no counts present
    return np.ones(len(all_stances), dtype=float) / len(all_stances)


def compute_cramers_v_for_issue(
    data: pd.DataFrame,
    issue: str,
    comparison_type: str,
    stance_col: str = "stance",
):
    """
    Compute Cramer's V for an issue comparing either models or languages.

    - If `comparison_type == 'culture'` → crosstab on 'culture' vs stance.
    - Else (including 'model') → crosstab on 'language' vs stance.
    """
    issue_data = data[data["new_id"] == issue]
    if len(issue_data) == 0:
        return None, None

    if comparison_type == "culture":
        contingency = pd.crosstab(issue_data["culture"], issue_data[stance_col])
    else:  # language branch (unchanged behavior)
        contingency = pd.crosstab(issue_data["language"], issue_data[stance_col])

    # Need at least 2x2 for chi-square / Cramer's V
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return None, None

    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    min_expected = expected.min()
    chi2_valid = min_expected >= 5

    # Cramer's V
    cv = cramers_v(contingency.values)

    return {
        "cramers_v": cv,
        "chi2_p_value": p_value,
        "chi2_statistic": chi2,
        "chi2_valid": chi2_valid,
        "min_expected_freq": min_expected,
        "contingency_shape": contingency.shape,
    }, contingency


# ── Main analytics ────────────
def analyze_issue_effects_cross_origin(
    data: pd.DataFrame,
    issues: Iterable[str],
    map_culture: Dict[str, List[str]],
    languages: Iterable[str],
    stance_col: str = "stance",
) -> pd.DataFrame:
    """
    Analyze model and language effects using CROSS-ORIGIN comparisons only.
    Only compares Western models with Chinese models (not within-origin).
    """
    results: List[Dict[str, object]] = []

    western_models = map_culture["Western"]
    chinese_models = map_culture["Chinese"]

    print("Computing per-issue effects analysis (CROSS-ORIGIN ONLY)...")
    print(f"Western models: {western_models}")
    print(f"Chinese models: {chinese_models}")

    for i, issue in enumerate(issues):
        print(f"Processing issue {i+1}/{len(issues)}: {issue}")

        result = {
            "new_id": issue,
            "model_mean_jsd": np.nan,
            "language_mean_jsd": np.nan,
            "jsd_difference": np.nan,  # model - language
            "jsd_p_value": np.nan,
            "jsd_test_statistic": np.nan,
            "jsd_effect_direction": "No Data",
            "jsd_significance": "No Data",
            "model_cramers_v": np.nan,
            "language_cramers_v": np.nan,
            "model_cramers_p_value": np.nan,
            "language_cramers_p_value": np.nan,
            "model_chi2_valid": False,
            "language_chi2_valid": False,
            "n_responses": 0,
            "n_model_jsds": 0,
            "n_language_jsds": 0,
        }

        # Issue subset
        issue_data = data[data["new_id"] == issue]
        result["n_responses"] = len(issue_data)

        if len(issue_data) == 0:
            results.append(result)
            continue

        # --- JSD: CROSS-ORIGIN MODELS ONLY (language fixed) ---
        model_jsds: List[float] = []
        for language in languages:
            for w_model in western_models:
                for c_model in chinese_models:
                    dist1 = get_stance_distribution_model(data, issue, w_model, language, stance_col)
                    dist2 = get_stance_distribution_model(data, issue, c_model, language, stance_col)

                    if dist1 is not None and dist2 is not None:
                        model_jsds.append(calculate_jsd(dist1, dist2))

        result["n_model_jsds"] = len(model_jsds)

        # --- Language JSDs (model fixed) ---
        language_jsds: List[float] = []
        all_models = western_models + chinese_models
        for model in all_models:
            for lang1, lang2 in combinations(languages, 2):
                dist1 = get_stance_distribution_model(data, issue, model, lang1, stance_col)
                dist2 = get_stance_distribution_model(data, issue, model, lang2, stance_col)
                if dist1 is not None and dist2 is not None:
                    language_jsds.append(calculate_jsd(dist1, dist2))

        result["n_language_jsds"] = len(language_jsds)

        # Means
        if model_jsds:
            result["model_mean_jsd"] = float(np.mean(model_jsds))
        if language_jsds:
            result["language_mean_jsd"] = float(np.mean(language_jsds))

        # Difference & tests
        if model_jsds and language_jsds:
            result["jsd_difference"] = result["model_mean_jsd"] - result["language_mean_jsd"]
            try:
                if len(model_jsds) >= 10 and len(language_jsds) >= 10:
                    print(f"Using t-test")
                    statistic, p_value = ttest_ind(model_jsds, language_jsds, alternative="two-sided")
                elif len(model_jsds) >= 3 and len(language_jsds) >= 3:
                    print(f"Using Mann–Whitney U test")
                    statistic, p_value = mannwhitneyu(model_jsds, language_jsds, alternative="two-sided")
                else:
                    print(f"Using Wilcoxon rank-sum test")
                    statistic, p_value = ranksums(model_jsds, language_jsds)

                result["jsd_p_value"] = float(p_value)
                result["jsd_test_statistic"] = float(statistic)

                if p_value < 0.05:
                    result["jsd_significance"] = "***"
                    result["jsd_effect_direction"] = (
                        "Model-Origin >> Language" if result["jsd_difference"] > 0 else "Language >> Model-Origin"
                    )
                elif p_value < 0.10:
                    result["jsd_significance"] = "*"
                    result["jsd_effect_direction"] = (
                        "Model-Origin > Language" if result["jsd_difference"] > 0 else "Language > Model-Origin"
                    )
                else:
                    result["jsd_significance"] = "ns"
                    result["jsd_effect_direction"] = "No difference"
            except Exception as e:
                print(f"  Warning: Could not perform JSD test for {issue}: {str(e)}")

        # --- Cramer's V ---
        model_stats, _ = compute_cramers_v_for_issue(data, issue, "model", stance_col)
        if model_stats:
            result["model_cramers_v"] = model_stats["cramers_v"]
            result["model_cramers_p_value"] = model_stats["chi2_p_value"]
            result["model_chi2_valid"] = model_stats["chi2_valid"]

        language_stats, _ = compute_cramers_v_for_issue(data, issue, "language", stance_col)
        if language_stats:
            result["language_cramers_v"] = language_stats["cramers_v"]
            result["language_cramers_p_value"] = language_stats["chi2_p_value"]
            result["language_chi2_valid"] = language_stats["chi2_valid"]

        results.append(result)

    return pd.DataFrame(results)


def format_results_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create a nicely formatted summary table (purely presentational)."""
    df = results_df.copy()

    numeric_cols = [
        "model_mean_jsd",
        "language_mean_jsd",
        "jsd_difference",
        "jsd_p_value",
        "jsd_test_statistic",
        "model_cramers_v",
        "language_cramers_v",
        "model_cramers_p_value",
        "language_cramers_p_value",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].round(4)

    def sig_marker(p: float) -> str:
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        if p < 0.10:
            return "."
        return ""

    df["model_cramers_sig"] = df.apply(lambda r: sig_marker(r.get("model_cramers_p_value")), axis=1)
    df["language_cramers_sig"] = df.apply(lambda r: sig_marker(r.get("language_cramers_p_value")), axis=1)

    column_order = [
        "new_id",
        "n_responses",
        "n_model_jsds",
        "n_language_jsds",
        "model_mean_jsd",
        "language_mean_jsd",
        "jsd_difference",
        "jsd_test_statistic",
        "jsd_p_value",
        "jsd_significance",
        "jsd_effect_direction",
        "model_cramers_v",
        "model_cramers_p_value",
        "model_cramers_sig",
        "model_chi2_valid",
        "language_cramers_v",
        "language_cramers_p_value",
        "language_cramers_sig",
        "language_chi2_valid",
    ]
    existing = [c for c in column_order if c in df.columns]
    return df[existing]


def create_summary_statistics(results_df: pd.DataFrame) -> Dict[str, float]:
    """Create simple summary statistics about the analysis (keys unchanged)."""
    summary: Dict[str, float] = {}
    summary["total_issues"] = len(results_df)
    summary["issues_with_data"] = len(results_df[results_df["n_responses"] > 0])

    jsd_sig = results_df["jsd_significance"].value_counts()
    summary["jsd_significant"] = jsd_sig.get("***", 0)
    summary["jsd_marginal"] = jsd_sig.get("*", 0)
    summary["jsd_nonsignificant"] = jsd_sig.get("ns", 0)

    summary["mean_model_jsds"] = results_df["n_model_jsds"].mean()
    summary["mean_language_jsds"] = results_df["n_language_jsds"].mean()

    summary["model_cramers_significant"] = len(results_df[results_df["model_cramers_p_value"] < 0.05])
    summary["language_cramers_significant"] = len(results_df[results_df["language_cramers_p_value"] < 0.05])

    summary["model_chi2_valid_count"] = int(results_df["model_chi2_valid"].sum())
    summary["language_chi2_valid_count"] = int(results_df["language_chi2_valid"].sum())

    return summary


def run_comprehensive_analysis_cross_origin(
    data: pd.DataFrame,
    map_culture: Dict[str, List[str]],
    stance_col: str = "stance",
) -> Optional[pd.DataFrame]:
    """
    Main function to run comprehensive per-issue analysis with CROSS-ORIGIN comparisons.
    Returns a formatted DataFrame with all metrics (printing preserved).
    """
    required_cols = ["new_id", "model", "language", stance_col]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("STARTING COMPREHENSIVE PER-ISSUE ANALYSIS (CROSS-ORIGIN ONLY)")
    print("=" * 80)

    original_len = len(data)
    clean_data = data.dropna(subset=["new_id", "model", "language", stance_col])
    print(f"Filtered {original_len - len(clean_data)} rows with missing values")
    print(f"Analyzing {len(clean_data)} complete responses")

    if len(clean_data) == 0:
        print("No complete data remaining after filtering!")
        return None

    issues = clean_data["new_id"].unique()
    languages = clean_data["language"].unique()

    print("\nDataset summary:")
    print(f"  {len(issues)} issues")
    print(f"  {len(languages)} languages")
    print(f"  Western models: {map_culture['Western']}")
    print(f"  Chinese models: {map_culture['Chinese']}")

    results_df = analyze_issue_effects_cross_origin(clean_data, issues, map_culture, languages, stance_col)
    formatted_df = format_results_table(results_df)
    summary = create_summary_statistics(results_df)

    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total issues analyzed: {summary['total_issues']}")
    print(f"Issues with data: {summary['issues_with_data']}")
    print("\nJSD Analysis (Cross-Origin Comparisons):")
    print(f"  Average model JSDs per issue: {summary['mean_model_jsds']:.1f}")
    print(f"  Average language JSDs per issue: {summary['mean_language_jsds']:.1f}")
    print(f"  Significant (p<0.05): {summary['jsd_significant']}")
    print(f"  Marginal (p<0.10): {summary['jsd_marginal']}")
    print(f"  Non-significant: {summary['jsd_nonsignificant']}")
    print("\nCramer's V Analysis:")
    print(f"  Model effects significant: {summary['model_cramers_significant']}")
    print(f"  Language effects significant: {summary['language_cramers_significant']}")
    print(f"  Valid chi-square tests (model): {summary['model_chi2_valid_count']}")
    print(f"  Valid chi-square tests (language): {summary['language_chi2_valid_count']}")
    print(f"\nAnalysis complete! DataFrame shape: {formatted_df.shape}")

    return formatted_df

