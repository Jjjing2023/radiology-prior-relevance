import json
import time
import requests
from datetime import datetime

with open('relevant_priors_public.json') as f:
    data = json.load(f)

truth_map = {(t['case_id'], t['study_id']): t['is_relevant_to_current'] for t in data['truth']}

correct = 0
total = 0
results_log = []

for i, case in enumerate(data['cases']): 
    payload = {"cases": [case]}
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=30)
        predictions = response.json()['predictions']
        for pred in predictions:
            truth = truth_map.get((pred['case_id'], pred['study_id']), False)
            if pred['predicted_is_relevant'] == truth:
                correct += 1
            total += 1
        if i % 50 == 0:
            print(f"Progress: {i}/996, Accuracy so far: {correct/total*100:.1f}%")
        time.sleep(0.5)  
    except Exception as e:
        print(f"Case {case['case_id']} failed: {e}")

# save the result
result = {
    "timestamp": datetime.now().isoformat(),
    "model": "gemini-2.0-flash",
    "total_cases": len(data['cases']),
    "total_predictions": total,
    "correct": correct,
    "accuracy": correct/total if total > 0 else 0
}

with open('eval_results.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f'\nFinal Accuracy: {correct}/{total} = {correct/total*100:.1f}%')
print(f'Results saved to eval_results.json')