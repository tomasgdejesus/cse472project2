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

# use ijson because neurips file is like 0.67 GB
data = []
with open('data/dataset_detection/gpt-4o-2024-08-06_neurips_train_2_grouped/gpt-4o-2024-08-06_neurips_train_grouped.json', 'r', encoding='UTF-8') as f:
    for candidate in ijson.items(f, 'item'):
        data.append({k: v for k, v in candidate.items() if k != 'content' and k != 'group_id'})

df = pd.DataFrame(data)

X = df.drop(columns=['label']).to_numpy()
encoder = LabelEncoder()
y = encoder.fit_transform(df['label'])

# use Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

# TODO: Use test data set, not the train data set

y_pred = clf.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
# Accuracy: 0.7749169435215947

# use RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
# Accuracy: 0.8745847176079734

# Accuracy is decent considering only using the features confidence, contribution, presentation, rating, and soudness
# RandomForest provides significantly better performance than Logistic Regression.