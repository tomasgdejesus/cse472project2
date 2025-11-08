import time
import ijson
import pandas as pd
import numpy as np
from itertools import permutations

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

trainset_path = 'data\\dataset_detection\\gpt-4o-2024-08-06_neurips_train_2_grouped\\gpt-4o-2024-08-06_neurips_train_grouped.json'
testset_path = 'data\\dataset_detection\\gpt-4o-2024-08-06_neurips_test_16_grouped\\gpt-4o-2024-08-06_neurips_test_grouped.json'

def main(task_num=5):
# Task 1
    # use ijson because neurips file is like 0.67 GB
    # TODO: use separate data[] for each of the tasks --- bc you will need to concat the datasets anyways for all the future tasks 
    data = []
    with open(trainset_path, 'r', encoding='UTF-8') as f:
        for candidate in ijson.items(f, 'item'):
            data.append({k: v for k, v in candidate.items() if k != 'content' and k != 'group_id'})

    df = pd.DataFrame(data)

    X_train = df.drop(columns=['label']).to_numpy()
    y_train = LabelEncoder().fit_transform(df['label'])

    data = []
    with open(testset_path, 'r', encoding='UTF-8') as f:
        for candidate in ijson.items(f, 'item'):
            data.append({k: v for k, v in candidate.items() if k != 'content' and k != 'group_id'})

    df = pd.DataFrame(data)

    X_test = df.drop(columns=['label']).to_numpy()
    y_test = LabelEncoder().fit_transform(df['label'])

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
    # Accuracy: 0.7511729222520107

    # use RandomForest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    # Accuracy: 0.8711461126005362

    # Accuracy is decent considering only using the features confidence, contribution, presentation, rating, and soudness
    # RandomForest provides significantly better performance than Logistic Regression.

    if task_num < 2:
        return
    
# Task 2

if __name__ == "__main__":
    main()