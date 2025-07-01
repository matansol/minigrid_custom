import pandas as pd
import numpy as np
from mysql_data_explorer import get_all_tables_from_mysql

def analyze_user_choices():
    """
    Analyze user choices and calculate correct answers for each similarity level
    """
    
    # Get all tables from MySQL
    print("🔄 Loading data from MySQL...")
    tables_dict = get_all_tables_from_mysql()
    
    if not tables_dict:
        print("❌ No data loaded")
        return
    
    # Check if required tables exist
    required_tables = ['user_choices', 'models_eval']
    for table in required_tables:
        if table not in tables_dict:
            print(f"❌ Required table '{table}' not found")
            return
    
    user_choices_df = tables_dict['user_choices']
    models_eval_df = tables_dict['models_eval']
    
    print(f"📊 User choices table: {user_choices_df.shape}")
    print(f"📊 Models eval table: {models_eval_df.shape}")
    
    # Display column names for debugging
    print(f"\n🔍 User choices columns: {list(user_choices_df.columns)}")
    print(f"🔍 Models eval columns: {list(models_eval_df.columns)}")
    
    # Merge user_choices with models_eval to get path values
    print("\n🔄 Merging data...")
    
    # First, let's see what the path columns look like
    print("\n📋 Sample user_choices data:")
    print(user_choices_df[['old_agent_path', 'new_agent_path', 'choice_to_update', 'simillarity_level']].head())
    
    print("\n📋 Sample models_eval data:")
    print(models_eval_df.head())
    
    # Assuming models_eval has columns like 'agent_path' and 'value' or similar
    # We need to merge based on the agent path to get the actual values
    
    # Let's check what columns are available in models_eval
    if 'agent_path' in models_eval_df.columns:
        path_col = 'agent_path'
    elif 'path' in models_eval_df.columns:
        path_col = 'path'
    else:
        print("❌ Cannot find agent path column in models_eval")
        print(f"Available columns: {list(models_eval_df.columns)}")
        return
    
    # Find the value column in models_eval
    value_cols = [col for col in models_eval_df.columns if 'value' in col.lower() or 'score' in col.lower() or 'reward' in col.lower()]
    if value_cols:
        value_col = value_cols[0]
        print(f"📊 Using '{value_col}' as the value column")
    else:
        print("❌ Cannot find value column in models_eval")
        print(f"Available columns: {list(models_eval_df.columns)}")
        return
    
    # Merge old agent path values
    old_merged = user_choices_df.merge(
        models_eval_df[[path_col, value_col]], 
        left_on='old_agent_path', 
        right_on=path_col, 
        how='left',
        suffixes=('', '_old')
    )
    
    # Merge new agent path values
    final_df = old_merged.merge(
        models_eval_df[[path_col, value_col]], 
        left_on='new_agent_path', 
        right_on=path_col, 
        how='left',
        suffixes=('_old', '_new')
    )
    
    print(f"\n📊 Merged data shape: {final_df.shape}")
    
    # Check for missing values
    missing_old = final_df[f'{value_col}_old'].isna().sum()
    missing_new = final_df[f'{value_col}_new'].isna().sum()
    print(f"❌ Missing old path values: {missing_old}")
    print(f"❌ Missing new path values: {missing_new}")
    
    # Remove rows with missing values
    final_df = final_df.dropna(subset=[f'{value_col}_old', f'{value_col}_new'])
    print(f"📊 Clean data shape: {final_df.shape}")
    
    # Calculate correct decisions
    print("\n🧮 Calculating correct decisions...")
    
    # A decision is correct if:
    # - choice_to_update = 1 AND new_value > old_value (user chose the better path)
    # - choice_to_update = 0 AND new_value <= old_value (user kept the better path)
    
    final_df['correct_decision'] = (
        ((final_df['choice_to_update'] == 1) & (final_df[f'{value_col}_new'] > final_df[f'{value_col}_old'])) |
        ((final_df['choice_to_update'] == 0) & (final_df[f'{value_col}_new'] <= final_df[f'{value_col}_old']))
    )
    
    # Calculate correct answers for each similarity level
    print("\n📈 Results by similarity level:")
    print("="*50)
    
    correct_answers_by_level = []
    
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

def main():
    """
    Main function to run the analysis
    """
    print("🎯 User Choices Analysis")
    print("="*50)
    
    try:
        correct_answers = analyze_user_choices()
        
        if correct_answers:
            print(f"\n✅ Analysis completed!")
            print(f"📊 Correct answers list: {correct_answers}")
        else:
            print("❌ Analysis failed")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 