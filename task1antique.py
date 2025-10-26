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

df = pd.read_json('data\dataset_detection\gpt-4o-2024-08-06_antique_train_1_grouped\gpt-4o-2024-08-06_antique_train_grouped.json')

# use Logistic Regression
# xi f = {f1, f2, f3}, and f is value of possible permutation
# yi is the label {0,1}, 0=Human, 1=LLM
X = np.array(df['ranking'].tolist())
encoder = LabelEncoder()
y = encoder.fit_transform(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
# Accuracy: 0.5520246027678114. Shows that ranking alone cannot be used by a simple classifier.

# use RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
# Accuracy: 0.5520246027678114. Shows that ranking alone cannot be used by a simple classifier.

# note that accuracy is same between both classifiers.

# TODO: maybe try permutation distance, i.e Kendall tau distance as feature xi. Email Dawei ab this