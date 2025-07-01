import pandas as pd
import numpy as np

def calculate_correct_answers_by_similarity_level(user_choices_df, models_eval_df):
    """
    Calculate correct answers for each similarity level (0-4)
    
    Args:
        user_choices_df: DataFrame with user_choices table
        models_eval_df: DataFrame with models_eval table
    
    Returns:
        list: Correct answers for each similarity level [level_0, level_1, level_2, level_3, level_4]
    """
    
    print("🔍 Analyzing user choices...")
    print(f"📊 User choices: {user_choices_df.shape}")
    print(f"📊 Models eval: {models_eval_df.shape}")
    
    # Find the path and value columns in models_eval
    path_col = None
    value_col = None
    
    # Look for path column
    for col in models_eval_df.columns:
        if 'path' in col.lower():
            path_col = col
            break
    
    # Look for value column
    for col in models_eval_df.columns:
        if any(keyword in col.lower() for keyword in ['value', 'score', 'reward', 'performance']):
            value_col = col
            break
    
    if not path_col:
        print(f"❌ Cannot find path column in models_eval")
        print(f"Available columns: {list(models_eval_df.columns)}")
        return None
    
    if not value_col:
        print(f"❌ Cannot find value column in models_eval")
        print(f"Available columns: {list(models_eval_df.columns)}")
        return None
    
    print(f"📊 Using '{path_col}' as path column and '{value_col}' as value column")
    
    # Merge data to get old and new path values
    print("🔄 Merging data...")
    
    # Merge old agent path values
    merged_df = user_choices_df.merge(
        models_eval_df[[path_col, value_col]], 
        left_on='old_agent_path', 
        right_on=path_col, 
        how='left',
        suffixes=('', '_old')
    )
    
    # Merge new agent path values
    final_df = merged_df.merge(
        models_eval_df[[path_col, value_col]], 
        left_on='new_agent_path', 
        right_on=path_col, 
        how='left',
        suffixes=('_old', '_new')
    )
    
    # Check for missing values
    missing_old = final_df[f'{value_col}_old'].isna().sum()
    missing_new = final_df[f'{value_col}_new'].isna().sum()
    
    if missing_old > 0 or missing_new > 0:
        print(f"⚠️  Missing values - Old: {missing_old}, New: {missing_new}")
        print("Removing rows with missing values...")
        final_df = final_df.dropna(subset=[f'{value_col}_old', f'{value_col}_new'])
        print(f"📊 Clean data shape: {final_df.shape}")
    
    # Calculate correct decisions
    print("🧮 Calculating correct decisions...")
    
    # A decision is correct if:
    # - choice_to_update = 1 AND new_value > old_value (user chose the better path)
    # - choice_to_update = 0 AND new_value <= old_value (user kept the better path)
    
    final_df['correct_decision'] = (
        ((final_df['choice_to_update'] == 1) & (final_df[f'{value_col}_new'] > final_df[f'{value_col}_old'])) |
        ((final_df['choice_to_update'] == 0) & (final_df[f'{value_col}_new'] <= final_df[f'{value_col}_old']))
    )
    
    # Calculate correct answers for each similarity level
    correct_answers_by_level = []
    
    print("\n📈 Results by similarity level:")
    print("="*40)
    
    for level in range(5):  # Similarity levels 0-4
        level_data = final_df[final_df['simillarity_level'] == level]
        
        if len(level_data) > 0:
            total_decisions = len(level_data)
            correct_decisions = level_data['correct_decision'].sum()
            accuracy = correct_decisions / total_decisions * 100
            
            print(f"🎚️ Level {level}: {correct_decisions}/{total_decisions} correct ({accuracy:.1f}%)")
            correct_answers_by_level.append(correct_decisions)
        else:
            print(f"🎚️ Level {level}: No data")
            correct_answers_by_level.append(0)
    
    print(f"\n📋 Correct answers list: {correct_answers_by_level}")
    return correct_answers_by_level

def analyze_user_choices_simple():
    """
    Simple function to get data and analyze user choices
    """
    try:
        from mysql_data_explorer import get_all_tables_from_mysql
        
        # Get data
        tables_dict = get_all_tables_from_mysql()
        
        if not tables_dict:
            print("❌ No data loaded")
            return None
        
        # Check required tables
        if 'user_choices' not in tables_dict:
            print("❌ user_choices table not found")
            return None
        
        if 'models_eval' not in tables_dict:
            print("❌ models_eval table not found")
            return None
        
        # Analyze
        return calculate_correct_answers_by_similarity_level(
            tables_dict['user_choices'], 
            tables_dict['models_eval']
        )
        
    except ImportError:
        print("❌ Cannot import mysql_data_explorer")
        print("💡 Make sure the file is in the same directory")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Example usage for Jupyter notebook:
"""
# In your Jupyter notebook:

# Method 1: Use the simple function
from user_choices_analysis import analyze_user_choices_simple
correct_answers = analyze_user_choices_simple()

# Method 2: Use the detailed function with your own data
from user_choices_analysis import calculate_correct_answers_by_similarity_level
from mysql_data_explorer import get_all_tables_from_mysql

tables_dict = get_all_tables_from_mysql()
correct_answers = calculate_correct_answers_by_similarity_level(
    tables_dict['user_choices'], 
    tables_dict['models_eval']
)

print(f"Correct answers by similarity level: {correct_answers}")
""" 