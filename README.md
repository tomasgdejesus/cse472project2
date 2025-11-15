# CSE 472 Project 2

Tomas Gabriel De Jesus

1229235672

## Requirements 

pip modules:
- scikit-learn
- pandas
- numpy
- ijson 

## File Structure
- run.py
    - Entry point for script execution
- task_\[name\].py
    - Program for tasks specific to the named dataset

## Internal TODO
- Get API key for CreateAI or some LLM/NLP classifier to add custom features
- Calculate F1 Scores and accuracy (Task 1, ...)
- Move onto group-level detection
- TODO: For Task 2, save llm/linguistic feature corresponding to the sample's query/responses. Then, for the train/test data set, 
when concatenating, I need to fill in the corresponding features for each sample if it is missing the llm/linguistic feature data.
    - Structure: llm_feature_score_map, k = query: str, v = docs: dict
        - docs, k = doc_n, v = avg llm score of doc_n