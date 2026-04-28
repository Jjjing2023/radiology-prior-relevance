import json
import requests

with open('relevant_priors_public.json') as f:
    data = json.load(f)

truth_map = {(t['case_id'], t['study_id']): t['is_relevant_to_current'] for t in data['truth']}

correct = 0
total = 0

# 把前10个case发给API
payload = {"cases": data['cases'][:10]}
response = requests.post("http://127.0.0.1:8000/predict", json=payload)
predictions = response.json()['predictions']

for pred in predictions:
    truth = truth_map.get((pred['case_id'], pred['study_id']), False)
    if pred['predicted_is_relevant'] == truth:
        correct += 1
    total += 1

print(f'Accuracy: {correct}/{total} = {correct/total*100:.1f}%')