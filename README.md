# Cross-Lingual Stance Analysis of U.S.- and China-related Issues in LLM Responses

This repository accompanies the paper **"Bias in the East, Bias in the West: A Bilingual Analysis of LLM Political Bias on U.S. and Chinese Issues"** .  
It contains all code and analysis pipelines used for data collection, model response generation, and statistical evaluation.

---

## Overview

This project investigates how **language** and **model origin** influence stance distributions in responses to U.S.–China–related issues.  
The workflow spans **data scraping**, **issue identification**, **prompt generation**, **model querying**, **evaluation**, and **statistical analysis**.

All analyses are implemented in Python and follow a modular pipeline organized by stage.

---

## Project Structure

```bash
01_scrape_data/ # Web scrapers for U.S. and Chinese media sources
├── abc/ #from american media
│ └── abc_scraper.py
├── american_news/
│ └── american_news_scraper.py # from chinese media
├── axios/
│ └── axios_scraper.py #from american media
├── cnn/ 
│  └── cnn_scraper.py #from american media
├── fox/
│ └── fox_scraper.py #from american media
├── nbc/
│ └── nbc_scraper.py #from american media
├── npr/
│ └── npr_scraper.py #from american media
└── politico/
  └──politico_scraper.py #from american media

02_issue_identification/ # Issue clustering & keyword extraction
├── chinese/
│ ├── a_clustering.py
│ ├── b_chinese_tfidf.py
│ ├── c_sample_chinese_issues.py
│ └── chinese_stopwords.txt
├── english/
│ ├── a_clustering.py
│ └── b_tfidf.py
└── cluster_file.py

03_template/ # Prompt template generation
└── a_generate_prompts.py

04_model_response/ # Query LLMs and collect responses
└── generate_model_response.py

05_evaluation/ # Judge & score model responses
├── a_generate_instructions.py
└── b_generate_judge_model_response.py

06_data_analysis/ # Statistical analysis & visualization
├── figs/ # Figures for the paper
├── scripts/ # Utility functions (e.g., cramers_v, jsd)
├── a_rename_dataset.ipynb
├── b_analyze_distribution.ipynb
├── c_analyze_model_similarities.ipynb
├── d_analyze_relative_effects.ipynb
└── e_statistical_sig_tests.ipynb
```