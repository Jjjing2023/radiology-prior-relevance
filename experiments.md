# Experiments

## Baseline

Started with a rule-based approach using keyword matching on `study_description` to extract modality (MRI, CT, XR, etc.) and body part (BRAIN, CHEST, etc.). Required both same modality AND same body part to predict relevant.

- Local accuracy: 82.0%

## What Worked

**Removing modality constraint:** Analysis of the labeled data showed that same body part + different modality cases are relevant 82.5% of the time. Removing the modality requirement and predicting relevant based on body part alone better reflects clinical practice.

**Adding LLM (Gemini 2.0 Flash):** Replaced rule-based logic with batched LLM inference. All prior studies for a case are sent in a single prompt to avoid timeout issues on cases with 100+ priors. The LLM understands medical terminology and synonyms (e.g. HEAD = BRAIN) that keyword matching misses.

- Local accuracy (50 cases): 85.1%
- Quick API Check score: 91.33%

**Caching:** Added MD5-based caching (keyed on case_id + current description + prior descriptions) to avoid redundant LLM calls for repeated study pairs, reducing latency and API costs.

**Rule-based fallback:** Added rule-based prediction as a fallback when the LLM call fails (e.g. API rate limit, timeout, malformed JSON response). This ensures the endpoint always returns a valid prediction instead of an HTTP 500 error, improving reliability and preventing skipped predictions from counting as incorrect.

**LLM hardening:** Added retry logic (up to 3 attempts) and a check that every prior study received a prediction before returning, defaulting missing predictions to false.

## What Failed

**Modality restriction:** Requiring same modality reduced accuracy because same body part across different modalities (e.g. CT and MRI of the brain) is still clinically relevant. Radiologists regularly compare across modalities for the same region.

**Rule-based alone is insufficient:** Keyword matching cannot handle complex medical abbreviations and shorthand (e.g. "NM myo perf SPECT" for a cardiac nuclear medicine study, or "MAM diagnostic RT with tomo" for a mammogram). Many study descriptions failed to match any body part keyword, causing incorrect predictions.

## How I Would Improve It

- **Train a non-LLM classifier:** The labeled dataset (27,614 examples) is large enough to train a dedicated binary classifier on study description pairs — for example, a TF-IDF + logistic regression baseline or a fine-tuned BioBERT model. This would likely match or exceed the LLM's accuracy while being significantly cheaper, faster, and fully reproducible without external API dependencies. The prompt-based approach was chosen for speed of iteration, but a trained classifier would be more production-appropriate.
- **Better medical NLP for fallback:** Replace keyword matching with a medical NLP model that understands abbreviations and clinical terminology (e.g. "NM myo perf SPECT" → cardiac nuclear medicine) for a more reliable fallback.
- **Incorporate study date as a hard signal:** More recent priors are generally more relevant. Currently the prompt only suggests the model consider dates, which could be enforced more explicitly.
- **Persistent caching:** Current cache is in-memory and resets on service restart. Using Redis would preserve cache across restarts and reduce repeated API calls.
- **Ranking instead of binary filtering:** Rather than returning a binary relevant/not-relevant decision, a ranked list of priors by relevance score would give radiologists more flexibility to decide how many priors to review.

## Radiologist Workflow Considerations

**Latency:** The current implementation makes one batched LLM call per case, completing in ~1-2 seconds for typical cases. This is acceptable for pre-fetch scenarios (e.g. predicting relevant priors before the radiologist opens the study), but may be too slow for real-time inline display during reading. A trained classifier would reduce latency to milliseconds.

**Ranking vs. binary filtering:** The current system returns a binary true/false for each prior. In practice, radiologists may benefit more from a ranked list. For example, "most recent same-modality same-body-part study" ranked above "older cross-modality study." Binary filtering risks surfacing too many or too few priors depending on the threshold.

**Impact of errors on reading efficiency:** False negatives (missing a relevant prior) are more costly than false positives (showing an irrelevant one) — a missed prior could cause a radiologist to overlook disease progression or misinterpret a finding. False positives add cognitive load but are less dangerous. This asymmetry suggests optimizing for recall over precision when tuning the decision threshold.
