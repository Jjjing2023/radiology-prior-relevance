from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
import numpy as np

with open('relevant_priors_public.json') as f:
    data = json.load(f)

truth_map = {(t['case_id'], t['study_id']): t['is_relevant_to_current'] for t in data['truth']}

# build trining data
X, y = [], []
for case in data['cases']:
    current_desc = case['current_study']['study_description']
    for prior in case['prior_studies']:
        #cat current + prior description as features
        X.append(current_desc + " [SEP] " + prior['study_description'])
        y.append(truth_map.get((case['case_id'], prior['study_id']), False))

# TF-IDF + Logistic Regression
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_vec, y)
preds = clf.predict(X_vec)
print(f"TF-IDF + LR Accuracy: {accuracy_score(y, preds)*100:.1f}%")