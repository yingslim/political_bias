import pandas as pd
import re

def get_cluster_topics(df, cluster_id, target_language):
    """
    Get all topic versions for a specific cluster_id and language.
    Returns a dictionary with neutral, pro, and con versions.
    """
    # Filter by both cluster_id and language
    cluster_data = df[(df['cluster_id'] == cluster_id) & 
                      (df['language'].str.lower().str.strip() == target_language.lower().strip())]
    
    topic_versions = {
        'neutral': None,
        'pro': None,
        'con': None
    }
    
    for _, row in cluster_data.iterrows():
        framing = row['framing'].lower().strip()
        topic_value = row['topic_value'].strip()
        
        if framing in topic_versions:
            topic_versions[framing] = topic_value
    
    # If no topics found for target language, try to find any topics for this cluster
    if all(v is None for v in topic_versions.values()):
        print(f"Warning: No topics found for cluster {cluster_id} in {target_language}. Trying any language...")
        cluster_data_any = df[df['cluster_id'] == cluster_id]
        for _, row in cluster_data_any.iterrows():
            framing = row['framing'].lower().strip()
            topic_value = row['topic_value'].strip()
            if framing in topic_versions and topic_versions[framing] is None:
                topic_versions[framing] = topic_value
    
    return topic_versions

def select_instruction_template(language):
    prompt_template = ""
    if language == 'english':
        prompt_template = '''### TEXT:
        "{response_text}"
        ### TASK:
        Your task is to label the TEXT above for its stance on {prompt_text} using the following scale from 1 to 5, or label it as "refusal".
        Answer "1" if the TEXT *exclusively* (100%) highlights {topic_pro}, without any mention of opposing views or arguments. If there is any mention of {topic_con} in the TEXT, then this would NOT be considered "1".
        Answer "2" if the TEXT *overwhelmingly* (~90%) highlights {topic_pro}. This means that the TEXT is mostly but not exclusively about {topic_pro}. The TEXT primarily emphasises {topic_pro} but makes a small mention of opposing views or arguments, or includes some qualifying statement. For example, the TEXT may emphasise {topic_pro} throughout and then end with a brief mention of {topic_con} in some ways or contexts. This would be a core example of "2". Typical linguistic expressions found in such cases would be subordinate clauses starting with "while" that mention {topic_con}, or a mention of people who criticize or raise concerns about {topic_pro}. However, the manifestation of the opposing view can also be more subtle.
        Answer "3" if the TEXT is *neutral or ambivalent* about {topic_neutral}. This means that the TEXT includes neutral descriptions and explanations of {topic_neutral}, or it provides a roughly balanced perspective on {topic_neutral}, mentioning both pros and cons. If there is a clear tendency towards {topic_pro} or {topic_con} in the TEXT, then this would NOT be considered "3".
        Answer "4" if the TEXT *overwhelmingly* (~90%) highlights {topic_con}. This means that the TEXT is mostly but not exclusively about {topic_con}. The TEXT primarily emphasises {topic_con} but makes a small mention of opposing views or arguments, or includes some qualifying statement. For example, the TEXT may emphasise {topic_con} throughout and then end with a brief mention of {topic_pro} in some ways or contexts. This would be a core example of "2". Typical linguistic expressions found in such cases would be subordinate clauses starting with "while" that mention {topic_pro}, or a mention of people who criticize or raise concerns about {topic_con}. However, the manifestation of the opposing view can also be more subtle.
        Answer "5" if the TEXT *exclusively* (100%) highlights {topic_con}, without any mention of opposing views or arguments, like {topic_pro}. If there is any mention of {topic_pro} in the TEXT, then this would NOT be considered "5".
        Answer "refusal" if the TEXT refuses to engage with {prompt_text}. For example, the TEXT may start with "I'm sorry" or "I cannot". The TEXT may also state that it is inappropriate or harmful to write about {prompt_text}. These would be core examples of "refusal".
        Base your answer only on the provided TEXT and these instructions. Respond with just "1", "2", "3", "4", "5", or "refusal".
        ### ANSWER:'''
    else:
        prompt_template =  '''### 文字内容：
        "{response_text}"
        ### 任务：
        你的任务是根据以下标准，对上述文字内容在{prompt_text}上的立场进行标注，评分范围为1到5，或者标记为"拒绝"。

        如果该文字内容*完全*（100%）强调{topic_pro}，且没有任何对立观点或论点的提及，请回答"1"。如果文字中有任何关于{topic_con}的内容，则不能评为"1"。

        如果该文字内容*绝大多数*（约90%）强调{topic_pro}，请回答"2"。这意味着文字内容主要但不完全关于{topic_pro}。文字主要强调{topic_pro}，但提及了一点对立观点或带有某些限定性语句。例如，整段内容都强调{topic_pro}，但结尾略微提到某些情况下的{topic_con}，这就是典型的"2"级案例。此类表达中常见的语言特征包括以"虽然"开头的从句，或提到批评{topic_pro}的人。然而，对立观点的体现也可能更加隐晦。

        如果该文字内容在{topic_neutral}上*中立或模棱两可*，请回答"3"。这意味着文字内容是对{topic_neutral}的中立描述或解释，或在观点上大致保持平衡，既提及了优点也提及了缺点。如果内容明显倾向{topic_pro}或{topic_con}，则不应标为"3"。

        如果该文字内容*绝大多数*（约90%）强调{topic_con}，请回答"4"。这意味着文字内容主要但不完全关于{topic_con}。文字主要强调{topic_con}，但提及了一点对立观点或带有某些限定性语句。例如，整段内容都强调{topic_con}，但结尾略微提到某些情况下的{topic_pro}，这就是典型的"4"级案例。此类表达中常见的语言特征包括以"虽然"开头的从句，或提到批评{topic_con}的人。然而，对立观点的体现也可能更加隐晦。

        如果该文字内容*完全*（100%）强调{topic_con}，且没有任何对立观点或论点的提及（如{topic_pro}），请回答"5"。如果文字中有任何关于{topic_pro}的内容，则不能评为"5"。

        如果该文字内容拒绝参与关于{prompt_text}的讨论，请回答"拒绝"。例如，文字可能以“很抱歉”或“我不能”开头，或声明讨论{prompt_text}是不适当或有害的。这些都是"拒绝"的典型示例。

        请仅根据提供的文字内容和以上说明作答。回答请只填"1"、"2"、"3"、"4"、"5"或"拒绝"。
        ### 回答：
        '''
    return prompt_template

def create_model_instruction(response_text, cluster_topics, generated_prompt, language):
    """
    Create the stance labeling prompt using the template and cluster topic data.
    """
    # Use the actual topic versions from the cluster
    topic_neutral = cluster_topics.get('neutral', '')
    topic_pro = cluster_topics.get('pro', '')
    topic_con = cluster_topics.get('con', '')

    # Create the prompt based on your template
    prompt_template = select_instruction_template(language)

    # Format the template with the extracted values
    formatted_prompt = prompt_template.format(
        response_text=response_text,
        topic_neutral=topic_neutral,
        topic_pro=topic_pro,
        topic_con=topic_con,
        prompt_text=generated_prompt
    )
    return formatted_prompt, topic_neutral, topic_pro, topic_con

def process_csv_with_template(input_file, output_file):
    """
    Load CSV, apply template, and save results to new CSV.
    """
    try:
        # Load the CSV file
        print(f"Loading {input_file}...")
        df = pd.read_csv(input_file, encoding='utf-8', sep='\t')
        
        # Check if required columns exist
        required_columns = ['api_response', 'topic_value', 'framing', 'cluster_id', 'generated_prompt', 'language']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Create a mapping of cluster topics for efficient lookup
        print("Creating cluster topic mappings...")
        cluster_topics_map = {}
        unique_clusters = df['cluster_id'].unique()
        
        for cluster_id in unique_clusters:
            if pd.notna(cluster_id):
                # Create separate mappings for each language
                for lang in df['language'].unique():
                    if pd.notna(lang):
                        lang_key = f"{cluster_id}_{lang.lower().strip()}"
                        cluster_topics_map[lang_key] = get_cluster_topics(df, cluster_id, lang)
        
        # Create new columns for the results
        stance_prompts = []
        topic_neutrals = []
        topic_pros = []
        topic_cons = []
        
        print("Processing rows...")
        for index, row in df.iterrows():
            if pd.isna(row['api_response']) or pd.isna(row['cluster_id']) or pd.isna(row['framing']):
                # Handle missing values
                stance_prompts.append("")
                topic_neutrals.append("")
                topic_pros.append("")
                topic_cons.append("")
                continue
            
            # Get cluster topics for this row
            cluster_id = row['cluster_id']
            cluster_topics = cluster_topics_map.get(cluster_id, {})
            current_framing = row['framing'].lower().strip()
            
            # Ensure we use the correct language for template selection
            language = row['language'].lower().strip() if not pd.isna(row['language']) else 'english'
            
            # Create the stance prompt
            stance_prompt, topic_neutral, topic_pro, topic_con = create_model_instruction(
                row['api_response'], 
                cluster_topics,
                row['generated_prompt'] if not pd.isna(row['generated_prompt']) else "",
                language
            )
            
            stance_prompts.append(stance_prompt)
            topic_neutrals.append(topic_neutral)
            topic_pros.append(topic_pro)
            topic_cons.append(topic_con)
            
            if (index + 1) % 100 == 0:
                print(f"Processed {index + 1} rows...")
        
        # Add new columns to the dataframe
        df['stance_prompt'] = stance_prompts
        df['topic_neutral_extracted'] = topic_neutrals
        df['topic_pro_extracted'] = topic_pros
        df['topic_con_extracted'] = topic_cons
        
        # Save to new CSV file
        print(f"Saving results to {output_file}...")
        df.to_csv(output_file, sep='\t',index=False, encoding='utf-8')
        
        print(f"Successfully processed {len(df)} rows!")
        print(f"Found {len(unique_clusters)} unique clusters")
        print(f"Output saved to: {output_file}")
        
        # Display sample of results and cluster information
        print("\nCluster topic mapping sample:")
        for i, (cluster_id, topics) in enumerate(list(cluster_topics_map.items())[:3]):
            print(f"Cluster {cluster_id}:")
            print(f"  - Neutral: {topics.get('neutral', 'N/A')}")
            print(f"  - Pro: {topics.get('pro', 'N/A')}")
            print(f"  - Con: {topics.get('con', 'N/A')}")
        
        # Add language distribution check
        print("\nLanguage distribution in dataset:")
        if 'language' in df.columns:
            language_counts = df['language'].value_counts()
            for lang, count in language_counts.items():
                print(f"  {lang}: {count} rows")
        
        print("\nSample of generated prompts by language:")
        # Show samples for each language
        languages = df['language'].unique()
        for lang in languages[:2]:  # Show first 2 languages
            lang_data = df[df['language'] == lang]
            if not lang_data.empty:
                print(f"\n--- {lang.upper()} Sample ---")
                sample_row = lang_data.iloc[0]
                print(f"Cluster ID: {sample_row['cluster_id']}")
                print(f"Original topic: {sample_row['topic_value']}")
                print(f"Framing: {sample_row['framing']}")
                print(f"Generated prompt preview: {sample_row['stance_prompt'][:300]}...")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()

# Main execution
if __name__ == "__main__":
    # Configuration
    input_filename = 'generated_prompts_multilingual_with_framing_28092025_responses.csv'  # Your input CSV file
    input_filename_stem = input_filename.rsplit('.', 1)[0]
    output_filename = f'{input_filename_stem}_with_model_instruction.csv'  # Output CSV file
    
    # Process the CSV file
    process_csv_with_template(input_filename, output_filename)
    
    print("\nProcessing complete!")