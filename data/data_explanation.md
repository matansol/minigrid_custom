# Data Explanation

The data described in this document was collected as part of the DPU (Demonstrating Policy Updates) study using a Qualtrics survey platform. After preprocessing the raw survey responses, the dataset was organized into two main types of records: user-level data and choice-level data. This file describe each features in the data.

## `results/dpu_results_users.csv`

This file contains aggregated results for each user who participated in the DPU experiment.

### User Demographics & Grouping
- **user_id**: A unique identifier for each participant.
- **group** The experimental condition the user was assigned to - [Control, Same, Random, Salient-Contrast]
- **age**: The age of the user.
- **gender**: The gender of the user.
- **education**: The user's level of education.

### User Choices & Performance
- **number_of_choices**: The total number of decisions the user made.
- **number_of_accepts**: How many times the user accepted the agent's suggestion.
- **number_of_correct_choices_mean**: The count of user choices that were optimal based on the agent's generalized performance.
- **number_of_correct_choices_feedback**: The count of user choices that were optimal for the specific task shown (local performance).
- **number_of_improved_choices**: The number of times the user's update resulted in a better agent.
- **user_performance**: The user's overall score in the game.

### Final Agent & Trust
- **trust_the_agent**: A binary value (1/0) indicating if the user chose to use the final agent.
- **final_agent_name**: The name of the agent the user ended up with.
- **final_agent_rank**: The performance rank of the final agent compared to all other agents.
- **final_agent_mean_score**: The generalized performance score of the user's final agent.
- **final_agent_final_part_score**: The local performance score of the final agent on the specific task.
- **final_improved**: A binary value indicating if the final agent was better than the initial one.
- **evaluate_final_agent**: The user's rating of the final agent's quality.

### Surveys & Attitudes - User answers on the likart-scal 1-7
- **AI_1**: User's agreement with the statement "I believe AI will improve my life."
- **AI_2**: User's agreement with the statement "I believe that AI will improve my work."
- **AI_3**: User's agreement with the statement "I think I will use AI technology in the future."
- **AI_4**: User's agreement with the statement "I think AI technology is a threat to humans."
- **AI_5**: User's agreement with the statement "I think AI technology is positive for humanity."
- **ES_understandable**: User's agreement with the statement "The explanations of the agent’s paths were clear and easy to understand"
- **ES_overwhelming**: User's agreement with the statement "The explanations felt too detailed and overwhelming".
- **ES_feedback_contribution**: User's agreement with the statement "The explanations helped them see how their feedback changed the agent’s behavior".
- **ES_helpful**: User's agreement with the statement "The visual demonstrations of policy changes helped them understand the differences between agents".
- **ES_combined**: A combined score representing overall satisfaction with the system's explanations (mean of - ES_understandable, reversed[ES_overwhelming], ES_feedback_contribution and ES_helpful).
- **SS_2**: An attention-check question ("Please select 'Strongly agree'").

---

## `results/dpu_results_choices.csv`

This file contains detailed data for each individual choice made by users during the experiment.

### Identifiers & Context
- **user_id**: A unique identifier for each participant.
- **episode**: The trial or round number for the choice.
- **group**: The experimental condition of the user [Control, Same, Random, Salient-Contrast]
- **feedback_board_index**: Identifier for the specific environment board the user provide feedback on.
- **demo_board_index**: Identifier for the demonstration board, if shown.

### Agent & Scores
- **prev_agent**: The agent's name before the update.
- **updated_agent**: The agent's name after the update.
- **prev_agent_mean_score**: The generalized score (mean score on all the boards) of the previous agent.
- **update_agent_mean_score**: The generalized score (mean score on all the boards) of the updated agent.
- **prev_agent_feedback_score**: The local score (score on the feedback board) of the previous agent.
- **updated_agent_feedback_score**: The local score (score on the feedback board) of the updated agent.

### User Action & Outcomes
- **choice_to_update**: A binary value indicating if the user chose to update the agent.
- **choice_time**: The time in seconds the user took to make a decision.
- **correct_choice_generalized**: A binary value indicating if the user's choice was optimal based on the agent's generalized score.
- **correct_choice_feedback**: A binary value indicating if the user's choice was optimal for the local score.
- **improved_from_base**: Whether the updated agent is better than the very first agent (based on mean score).
- **improved_from_base_last**: Whether the updated agent is better than the very first agent (based on local score).
- **improved_from_prev_mean**: Whether the updated agent's mean score is better than the previous agent's.
- **improved_from_prev_feedback**: Whether the updated agent's local score is better than the previous agent's.

### Final State (repeated for each choice)
- **final_agent**: The name of the agent the user ultimately selected at the end of all episodes.
- **final_agent_mean_score**: The mean score of that final agent.
- **trust_the_agent**: The user's final decision to trust the agent, repeated for each row.
