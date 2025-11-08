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

# X structure:
# x1 = ranking vector
# x2 = llm_rank vector (llm_enhanced_feature.Ranking)
# x3 = llm Response1 Score (llm_enhanced_feature.Response1 Score)
# x4 = llm Response2 Score (llm_enhanced_feature.Response1 Score)
# x5 = llm Response3 Score (llm_enhanced_feature.Response1 Score)
def task2_features(df):
    ret = []
    for k, v in df.iterrows():
        llm_enhanced_feature = v['llm_enhanced_feature']
        print(llm_enhanced_feature)
        if llm_enhanced_feature is dict:
            ret.append(
                v['ranking'] +
                llm_enhanced_feature['Ranking'] +
                [
                    llm_enhanced_feature['Response1 Score'], 
                    llm_enhanced_feature['Response2 Score'],
                    llm_enhanced_feature['Response3 Score']
                ]
            )
        else:
            ret.append( ### TODO: Placeholder; fix features if they are missing
                v['ranking'] +
                [-1, -1, -1] +
                [
                    -1, 
                    -1,
                    -1
                ]
            )

    return np.array(ret)


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

    df_llm_train = pd.concat([df_train, pd.read_json(llm_enhanced_trainset_path)], ignore_index=True)
    llm_X_train = task2_features(df_llm_train)
    llm_y_train = encoder.transform(df_llm_train['label'])

    df_llm_test = pd.concat([df_test, pd.read_json(llm_enhanced_testset_path)], ignore_index=True)
    llm_X_test = task2_features(df_llm_test)
    llm_y_test = encoder.transform(df_llm_test['label'])

    clf = LogisticRegression()
    clf.fit(llm_X_train, llm_y_train)

    y_pred = clf.predict(llm_X_test)
    print("Logistic Regression Accuracy:", accuracy_score(llm_y_test, y_pred))

    # use RandomForest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(llm_X_train, llm_y_train)

    y_pred = clf.predict(llm_X_test)
    print("Random Forest Accuracy:", accuracy_score(llm_y_test, y_pred))

    # Logistic Regression Accuracy: 0.6670811666289849
    # Random Forest Accuracy: 0.6670811666289849
    # TODO: Fix overlapping

if __name__ == "__main__":
    main()