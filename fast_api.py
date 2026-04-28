from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from google import genai
import json
import hashlib
import os

load_dotenv()

app = FastAPI()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

cache = {}

class Study(BaseModel):
    study_id: str
    study_description: Optional[str] = ""
    study_date: Optional[str] = ""

class Case(BaseModel):
    case_id: str
    patient_id: Optional[str] = ""
    current_study: Study
    prior_studies: List[Study]

class Request(BaseModel):
    cases: List[Case]

class Prediction(BaseModel):
    case_id: str
    study_id: str
    predicted_is_relevant: bool

class Response(BaseModel):
    predictions: List[Prediction]

# ---- Rule-based fallback ----
def extract_modality_and_body_part(description: str):
    description = description.upper()
    modalities = ["MRI", "CT", "XR", "US", "PET", "NM", "MAM", "XRAY", "X-RAY", "SPECT", "DXA"]
    body_parts = ["BRAIN", "CHEST", "ABDOMEN", "PELVIS", "SPINE", "KNEE",
                  "HIP", "SHOULDER", "NECK", "LIVER", "HEART", "LUNG",
                  "BREAST", "CERVICAL", "LUMBAR", "THORACIC", "WRIST",
                  "ANKLE", "FOOT", "HAND", "FINGER", "FEMUR", "TIBIA"]
    found_modality = next((m for m in modalities if m in description), None)
    found_body = next((b for b in body_parts if b in description), None)
    return found_modality, found_body

def is_relevant(current_desc: str, prior_desc: str) -> bool:
    curr_modality, curr_body = extract_modality_and_body_part(current_desc)
    prior_modality, prior_body = extract_modality_and_body_part(prior_desc)
    
    # check just body part regardless of modality
    if curr_body and prior_body:
        return curr_body == prior_body
    
    return False

# ---- LLM prediction ----
def make_cache_key(case_id, current_desc, prior_descs):
    content = case_id + "|" + current_desc + "|" + ",".join(prior_descs)
    return hashlib.md5(content.encode()).hexdigest()

def llm_predict(case_id: str, current_study: Study, prior_studies: List[Study]) -> dict:
    cache_key = make_cache_key(
        case_id,
        current_study.study_description,
        [p.study_description for p in prior_studies]
    )
    if cache_key in cache:
        return cache[cache_key]

    prior_list = "\n".join([
        f"{i+1}. study_id={p.study_id}, description={p.study_description}, date={p.study_date}"
        for i, p in enumerate(prior_studies)
    ])

    prompt = f"""You are an expert radiologist. When reading a current examination, determine which prior examinations are worth reviewing for comparison.

Current examination: {current_study.study_description} (date: {current_study.study_date})

Prior examinations to evaluate:
{prior_list}

Rules for relevance:
1. RELEVANT: Same body part, regardless of modality (e.g. CT and MRI of same region are both relevant)
2. RELEVANT: HEAD = BRAIN, SPINE includes CERVICAL/THORACIC/LUMBAR, ABDOMEN includes LIVER/KIDNEY/PELVIS
3. NOT RELEVANT: Completely different body part (e.g. knee vs chest)

Return ONLY a valid JSON object mapping each study_id to true or false.
No markdown, no explanation, just JSON.
Example: {{"2453245": true, "992654": false}}"""

    # retry up to 3 times
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            result_text = response.text.strip()

            # remove markdown
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            result = json.loads(result_text)
            result = {str(k): bool(v) for k, v in result.items()}

            # verify every prior has a prediction
            for p in prior_studies:
                if str(p.study_id) not in result:
                    result[str(p.study_id)] = False

            cache[cache_key] = result
            return result

        except Exception as e:
            print(f"LLM attempt {attempt+1} failed: {e}")
            if attempt == 2:
                raise

    raise Exception("LLM failed after 3 attempts")

# ---- Endpoint ----
@app.post("/predict", response_model=Response)
def predict(request: Request):
    predictions = []
    for case in request.cases:
        current_desc = case.current_study.study_description or ""
        try:
            relevance_map = llm_predict(case.case_id, case.current_study, case.prior_studies)
            for prior in case.prior_studies:
                relevant = relevance_map.get(prior.study_id, False)
                predictions.append(Prediction(
                    case_id=case.case_id,
                    study_id=prior.study_id,
                    predicted_is_relevant=relevant
                ))
        except Exception as e:
            print(f"LLM failed for case {case.case_id}, falling back to rule-based: {e}")
            for prior in case.prior_studies:
                relevant = is_relevant(current_desc, prior.study_description or "")
                predictions.append(Prediction(
                    case_id=case.case_id,
                    study_id=prior.study_id,
                    predicted_is_relevant=relevant
                ))
    return Response(predictions=predictions)