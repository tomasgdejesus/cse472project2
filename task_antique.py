import time
import json 
import pandas as pd
import numpy as np
from itertools import permutations

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

trainset_path = 'data\\dataset_detection\\gpt-4o-2024-08-06_antique_train_1_grouped\\gpt-4o-2024-08-06_antique_train_grouped.json'
testset_path = 'data\\dataset_detection\\gpt-4o-2024-08-06_antique_test_1_grouped\\gpt-4o-2024-08-06_antique_test_grouped.json'

llm_enhanced_trainset_path = 'data\\features\\llm_enhanced_features\\ANTIQUE_train_Qwen3-8B.json'
llm_enhanced_testset_path = 'data\\features\\llm_enhanced_features\\ANTIQUE_test_Qwen3-8B.json'

def main(task_num=5):
    # use Logistic Regression
    # xi f = {f1, f2, f3}, and f is value of possible permutation
    # yi is the label {0,1}, 0=Human, 1=LLM

# Task 1
    encoder = LabelEncoder()
    df_train = pd.read_json(trainset_path)
    X_train = np.array(df_train['ranking'].tolist())
    y_train = encoder.fit_transform(df_train['label'])

    df_test = pd.read_json(testset_path)
    X_test = np.array(df_test['ranking'].tolist())
    y_test = encoder.transform(df_test['label'])

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
    # Accuracy: 0.5309479396303205. Shows that ranking alone cannot be used by a simple classifier.

    # use RandomForest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    # Accuracy: 0.5314566728845176. Shows that ranking alone cannot be used by a simple classifier.

    if task_num < 2:
        return

# Task 2

    llm_feature_score_map = {} # k = query: str, v = docs: dict (k = doc: str, v: pair (sum, count))
    # the score is stored as a pair (sum, count), so that the average score can be calculated

    # TODO: Code is ugly so it would be best to put these into functions

    # populate scores for llm_features
    df_llm_trainset = pd.read_json(llm_enhanced_trainset_path)
    for k, sample in df_llm_trainset.iterrows():
        sample_query = sample['query']
        sample_llm_enhanced_feature = sample['llm_enhanced_feature']
        if sample_query not in llm_feature_score_map:
            llm_feature_score_map[sample_query] = {}

        if sample_llm_enhanced_feature is None:
            continue

        docs_scores_map = llm_feature_score_map[sample_query]
        for i in range(3):
            cur_doc_score = sample_llm_enhanced_feature['Response' + str(i + 1) + ' Score']
            cur_doc = sample['docs'][i]
            if cur_doc in docs_scores_map:
                docs_scores_map[cur_doc] = (docs_scores_map[cur_doc][0] + cur_doc_score, docs_scores_map[cur_doc][1] + 1)
            else:
                docs_scores_map[cur_doc] = (cur_doc_score, 1)

    for k, sample in pd.read_json(llm_enhanced_testset_path).iterrows():
        sample_query = sample['query']
        sample_llm_enhanced_feature = sample['llm_enhanced_feature']
        if sample_query not in llm_feature_score_map:
            llm_feature_score_map[sample_query] = {}

        if sample_llm_enhanced_feature is None:
            continue

        docs_scores_map = llm_feature_score_map[sample_query]
        for i in range(3):
            cur_doc_score = sample_llm_enhanced_feature['Response' + str(i + 1) + ' Score']
            cur_doc = sample['docs'][i]
            if cur_doc in docs_scores_map:
                docs_scores_map[cur_doc] = (docs_scores_map[cur_doc][0] + cur_doc_score, docs_scores_map[cur_doc][1] + 1)
            else:
                docs_scores_map[cur_doc] = (cur_doc_score, 1)

    missing_doc_count = 0

    # apply matching llm_features to corresponding samples in train set
    X_train_task2 = []
    for k, sample in df_train.iterrows():
        sample_query = sample['query']
        if sample_query not in llm_feature_score_map:
            raise Exception("Query not in llm_feature_score_map")

        sample_docs = sample['docs']
        if sample_docs is None:
            raise Exception("Sample docs is None")
        
        llm_ranked_scores = []

        for i in range(3):
            cur_doc = sample_docs[i]
            cur_doc_score = -1 # signifies that the data set contains a response that has zero data for its llm generated judgement
            if cur_doc in llm_feature_score_map[sample_query]:
                cur_doc_score = llm_feature_score_map[sample_query][cur_doc][0] / llm_feature_score_map[sample_query][cur_doc][1]
            else:
                missing_doc_count += 1

            llm_ranked_scores.append(cur_doc_score)

        X_train_task2.append(sample['ranking'] + llm_ranked_scores)
    y_train_task2 = encoder.transform(df_train['label'])

    X_test_task2 = []
    for k, sample in df_test.iterrows():
        sample_query = sample['query']
        if sample_query not in llm_feature_score_map:
            raise Exception("Query not in llm_feature_score_map")

        sample_docs = sample['docs']
        if sample_docs is None:
            raise Exception("Sample docs is None")
        
        llm_ranked_scores = []

        for i in range(3):
            cur_doc = sample_docs[i]
            cur_doc_score = -1 # signifies that the data set contains a response that has zero data for its llm generated judgement
            if cur_doc in llm_feature_score_map[sample_query]:
                cur_doc_score = llm_feature_score_map[sample_query][cur_doc][0] / llm_feature_score_map[sample_query][cur_doc][1]
            else:
                missing_doc_count += 1

            llm_ranked_scores.append(cur_doc_score)

        X_test_task2.append(sample['ranking'] + llm_ranked_scores)
    y_test_task2 = encoder.transform(df_test['label']) 

    clf = LogisticRegression()
    clf.fit(X_train_task2, y_train_task2)

    y_pred = clf.predict(X_test_task2)
    print("Logistic Regression Accuracy:", accuracy_score(y_test_task2, y_pred))

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_task2, y_train_task2)

    y_pred = clf.predict(X_test_task2)
    print("Random Forest Classification Accuracy:", accuracy_score(y_test_task2, y_pred))
    print(f"Missing doc count: {missing_doc_count}")

if __name__ == "__main__":
    main()