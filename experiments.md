# Experiments

## Baseline

Started with a rule-based approach using keyword matching on `study_description` to extract modality (MRI, CT, XR, etc.) and body part (BRAIN, CHEST, etc.). Required both same modality AND same body part to predict relevant.

- Local accuracy: 82.0%

## What Worked

**Removing modality constraint:** Analysis of the labeled data showed that same body part + different modality cases are relevant 82.5% of the time. Removing the modality requirement and predicting relevant based on body part alone better reflects clinical practice.

**Adding LLM (Gemini 2.0 Flash):** Replaced rule-based logic with batched LLM inference. All prior studies for a case are sent in a single prompt to avoid timeout issues on cases with 100+ priors. The LLM understands medical terminology and synonyms (e.g. HEAD = BRAIN) that keyword matching misses.

- Local accuracy (10 cases): 83.5%
- Quick API Check score: 91.33%

**Caching:** Added MD5-based caching to avoid redundant LLM calls for repeated study pairs, reducing latency and API costs.

**Rule-based fallback:** Added rule-based prediction as a fallback when the LLM call fails (e.g. API rate limit, timeout, malformed JSON response). This ensures the endpoint always returns a valid prediction instead of an HTTP 500 error, improving reliability and preventing skipped predictions from counting as incorrect.

## What Failed

**Modality restriction:** Requiring same modality reduced accuracy because same body part across different modalities (e.g. CT and MRI of the brain) is still clinically relevant. Radiologists regularly compare across modalities for the same body part.

**Rule-based alone is insufficient:** Keyword matching cannot handle complex medical abbreviations and shorthand (e.g. "NM myo perf SPECT" for a cardiac nuclear medicine study, or "MAM diagnostic RT with tomo" for a mammogram). Many study descriptions failed to match any body part keyword, causing incorrect predictions.

## How I Would Improve It

- **Fine-tune a classifier:** Use the labeled dataset to train a dedicated binary classifier (e.g. fine-tuned BioBERT) on study description pairs, which would likely outperform a general-purpose LLM on this specific task.
- **Better medical NLP for fallback:** Replace keyword matching with a medical NLP model that understands abbreviations and clinical terminology for a more reliable fallback.
- **Incorporate study date as a hard signal:** More recent priors are generally more relevant. Currently the prompt only suggests the model consider dates, which could be enforced more explicitly in the logic.
- **Persistent caching:** Current cache is in-memory and resets on service restart. Using Redis would preserve cache across restarts and reduce repeated API calls.
