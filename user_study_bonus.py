import pandas as pd


def get_bonus(df, bonus, answers, questions=(), addons=()):
    df["Bonus"] = 0
    if questions:
        for q in questions:
            for add in addons:
                question = " ".join([q, add])
                df.loc[df[question] == answers[question], "Bonus"] += bonus
    else:
        for q in answers:
            df.loc[df[q] == answers[q], "Bonus"] += bonus
    df["Bonus"] = df["Bonus"].round(decimals=2)
    return df


df = pd.read_csv("results/DPU+Explanations+-+Prolific+-+BetweenSubject+-+Pilot_11_8_2025.csv")
date = "11/8/2025"
filtered_df = df[(pd.to_datetime(df["StartDate"]) > pd.to_datetime(date)) & (df["ProlificID"].astype(str).str.len() > 24)]
# correct_answers = answer_dict
bonus_fields = ["ProlificID", "Bonus"]

group_to_bonus = {"1": 0, "2": 0.5, "3": 1}
df["Bonus"] = df["bonus_level"].str[-1].map(group_to_bonus).fillna(0)
"""bonus"""
# df = get_bonus(df, bonus, correct_answers)
bonus_df = df[bonus_fields]
bonus_df.to_csv("bonus.csv", index=False)
