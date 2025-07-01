import pandas as pd
import numpy as np

def create_models_eval_dataframe(models_eval_list):
    """
    Convert the models_eval list to a DataFrame
    
    Args:
        models_eval_list: List of dictionaries with model paths and values
    
    Returns:
        pd.DataFrame: DataFrame with agent_path and value columns
    """
    data = []
    for item in models_eval_list:
        for path, value in item.items():
            # Extract the model name from the path
            # Assuming path format: "3,3,3,0.1,0.1Steps100Grid8_20250602"
            model_name = path
            data.append({
                'agent_path': model_name,
                'value': value
            })
    
    return pd.DataFrame(data)

def calculate_correct_answers_by_similarity_level(user_choices_df, models_eval_list):
    """
    Calculate correct answers for each similarity level (0-4)
    
    Args:
        user_choices_df: DataFrame with user_choices table
        models_eval_list: List of dictionaries with model paths and values
    
    Returns:
        list: Correct answers for each similarity level [level_0, level_1, level_2, level_3, level_4]
    """
    
    print("🔍 Analyzing user choices...")
    print(f"📊 User choices: {user_choices_df.shape}")
    print(f"📊 Models eval entries: {len(models_eval_list)}")
    
    # Convert models_eval list to DataFrame
    models_eval_df = create_models_eval_dataframe(models_eval_list)
    print(f"📊 Models eval DataFrame: {models_eval_df.shape}")
    
    # Display sample data
    print("\n📋 Sample user_choices data:")
    print(user_choices_df[['old_agent_path', 'new_agent_path', 'choice_to_update', 'simillarity_level']].head())
    
    print("\n📋 Sample models_eval data:")
    print(models_eval_df.head())
    
    # Extract model names from agent paths
    print("\n🔄 Processing agent paths...")
    
    def extract_model_name(path):
        """Extract model name from full path"""
        if pd.isna(path):
            return None
        # Extract the model folder name from the path
        # Example: "models/2,2,2,0,0.1Steps100Grid8_20250526/best_model.zip" -> "2,2,2,0,0.1Steps100Grid8_20250526"
        if 'models/' in str(path):
            parts = str(path).split('/')
            if len(parts) >= 2:
                return parts[1]  # Get the model folder name
        return str(path)
    
    # Create new columns with extracted model names
    user_choices_df['old_model_name'] = user_choices_df['old_agent_path'].apply(extract_model_name)
    user_choices_df['new_model_name'] = user_choices_df['new_agent_path'].apply(extract_model_name)
    
    print("\n📋 Sample processed data:")
    print(user_choices_df[['old_model_name', 'new_model_name', 'choice_to_update', 'simillarity_level']].head())
    
    # Merge data to get old and new path values
    print("\n🔄 Merging data...")
    
    # Merge old agent path values
    merged_df = user_choices_df.merge(
        models_eval_df, 
        left_on='old_model_name', 
        right_on='agent_path', 
        how='left',
        suffixes=('', '_old')
    )
    
    # Merge new agent path values
    final_df = merged_df.merge(
        models_eval_df, 
        left_on='new_model_name', 
        right_on='agent_path', 
        how='left',
        suffixes=('_old', '_new')
    )
    
    print(f"\n📊 Merged data shape: {final_df.shape}")
    
    # Check for missing values
    missing_old = final_df['value_old'].isna().sum()
    missing_new = final_df['value_new'].isna().sum()
    
    if missing_old > 0 or missing_new > 0:
        print(f"⚠️  Missing values - Old: {missing_old}, New: {missing_new}")
        print("Removing rows with missing values...")
        final_df = final_df.dropna(subset=['value_old', 'value_new'])
        print(f"📊 Clean data shape: {final_df.shape}")
    
    # Show sample of final data
    print("\n📋 Sample final data:")
    print(final_df[['old_model_name', 'new_model_name', 'value_old', 'value_new', 'choice_to_update', 'simillarity_level']].head())
    
    # Calculate correct decisions
    print("\n🧮 Calculating correct decisions...")
    
    # A decision is correct if:
    # - choice_to_update = 1 AND new_value > old_value (user chose the better path)
    # - choice_to_update = 0 AND new_value <= old_value (user kept the better path)
    
    final_df['correct_decision'] = (
        ((final_df['choice_to_update'] == 1) & (final_df['value_new'] > final_df['value_old'])) |
        ((final_df['choice_to_update'] == 0) & (final_df['value_new'] <= final_df['value_old']))
    )
    
    # Calculate correct answers for each similarity level
    correct_answers_by_level = []
    
    print("\n📈 Results by similarity level:")
    print("="*50)
    
    for level in range(5):  # Similarity levels 0-4
        level_data = final_df[final_df['simillarity_level'] == level]
        
        if len(level_data) > 0:
            total_decisions = len(level_data)
            correct_decisions = level_data['correct_decision'].sum()
            accuracy = correct_decisions / total_decisions * 100
            
            print(f"🎚️ Similarity Level {level}:")
            print(f"   📊 Total decisions: {total_decisions}")
            print(f"   ✅ Correct decisions: {correct_decisions}")
            print(f"   📈 Accuracy: {accuracy:.1f}%")
            print(f"   ❌ Incorrect decisions: {total_decisions - correct_decisions}")
            
            # Show some examples
            print(f"   📝 Examples:")
            examples = level_data[['old_model_name', 'new_model_name', 'value_old', 'value_new', 'choice_to_update', 'correct_decision']].head(3)
            for idx, row in examples.iterrows():
                decision = "UPDATE" if row['choice_to_update'] == 1 else "KEEP"
                correct = "✅" if row['correct_decision'] else "❌"
                print(f"      {correct} {decision}: {row['old_model_name']} ({row['value_old']:.2f}) -> {row['new_model_name']} ({row['value_new']:.2f})")
            print()
            
            correct_answers_by_level.append(correct_decisions)
        else:
            print(f"🎚️ Similarity Level {level}: No data")
            correct_answers_by_level.append(0)
    
    print("📋 Summary - Correct answers by similarity level:")
    print(f"Level 0: {correct_answers_by_level[0]}")
    print(f"Level 1: {correct_answers_by_level[1]}")
    print(f"Level 2: {correct_answers_by_level[2]}")
    print(f"Level 3: {correct_answers_by_level[3]}")
    print(f"Level 4: {correct_answers_by_level[4]}")
    
    return correct_answers_by_level

# Example usage for your notebook:
"""
# Copy this code to your notebook:

# Your existing data
models_eval = [
    {'3,3,3,0.1,0.1Steps100Grid8_20250602': -3.584860000000017}, 
    {'3,3,4,0.2,0.05Steps50Grid8_20250604': -16.003919999999667}, 
    {'2,2,4,-4,0.1Steps50Grid8_20250617': 3.3411399999999576}, 
    {'-1,-1,4,0.2,0.1Steps70Grid8_20250625': -3.6720200000000305}, 
    {'-0.5,2,4,-3,0.1Steps50Grid8_20250612_good': 4.5956399999999}, 
    {'-0.5,3,4,0.2,0.1Steps50Grid8_20250616': 0.7418600000000011}, 
    {'-1,4,-1,0.2,0.1Steps60Grid8_20250618': -3.6132600000000124}, 
    {'-1,3,4,-3,0.1Steps60Grid8_20250618': 2.717980000000023}, 
    {'-0.5,3,4,-3,0.1Steps50Grid8_20250616': 4.7921799999998855},
    {'-1,3,4,0.2,0.2Steps50Grid8_20250617': 0.7431600000000003}
]

# Run the analysis
correct_answers = calculate_correct_answers_by_similarity_level(user_choises, models_eval)
print(f"Correct answers by similarity level: {correct_answers}")
""" 