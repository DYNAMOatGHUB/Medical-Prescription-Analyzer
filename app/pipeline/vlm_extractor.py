"""
app/pipeline/vlm_extractor.py
──────────────────────────────
Stage 2 & 3 Combined – Local VLM Extraction (Enhanced)
Uses a Local Vision-Language Model (Qwen2.5-VL via Ollama) to directly
read and structure handwritten prescriptions, understanding tabular
and spatial layouts natively without needing cloud API keys.

Key optimisations:
  • Two-pass prompting (system + user) for sharper instruction-following
  • Few-shot example baked into the prompt so the VLM knows the exact output format
  • Explicit bilingual header mapping (Tamil, Hindi, Telugu, Malayalam, Kannada)
  • Robust JSON extraction with regex fallback for markdown-wrapped responses
  • Retry logic with temperature escalation for resilience
"""

from typing import List, Optional
import json
import os
import re
import base64
from pydantic import BaseModel, Field
import ollama
from loguru import logger

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Schema
# ──────────────────────────────────────────────────────────────────────────────

class Medicine(BaseModel):
    name: str = Field(description="Full medicine/drug name as written on the prescription")
    dosage: str = Field(
        description=(
            "Strength or amount per single intake (e.g., '500mg', '10ml', '1 Tablet', '5mg/5ml'). "
            "If the prescription does not mention a strength, output 'Not specified'."
        )
    )
    quantity: Optional[str] = Field(
        description="Total quantity dispensed (e.g., '10 tablets', '1 bottle').",
        default=None,
    )
    period_of_intake: str = Field(
        description=(
            "Human-readable dosage schedule combining the grid numbers with their column meanings. "
            "Example: '1-0-1 (Morning, Night)' or '1-1-1 (Morning, Afternoon, Night)'."
        )
    )
    route: Optional[str] = Field(
        description="Route of administration (e.g., Oral, IV, Topical, Inhaled).",
        default=None,
    )
    duration: Optional[str] = Field(
        description="How long to take it (e.g., '5 days', '1 week', '2 weeks').",
        default=None,
    )
    instructions: Optional[str] = Field(
        description="Special instructions (e.g., 'After food', 'Before sleep', 'Empty stomach').",
        default=None,
    )

class PrescriptionAnalysis(BaseModel):
    medicines: List[Medicine] = Field(description="List of all extracted medicines.")
    doctor_notes: Optional[str] = Field(
        description="Any additional instructions, diagnosis, or notes from the doctor.",
        default=None,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a world-class medical prescription digitisation engine. \
You have been trained on thousands of Indian handwritten prescriptions \
across multiple languages (Tamil, Hindi, Telugu, Malayalam, Kannada, English). \
You NEVER hallucinate medicine names or dosages — if you cannot read something clearly, \
you mark it with a '?' suffix (e.g., 'Amoxicillin?').

Your ONLY job is to return a single, valid JSON object. No prose, no explanation, no markdown."""

USER_PROMPT = """\
Carefully examine the attached prescription image.

## STEP 1 — Identify the Grid / Table Structure
Indian prescriptions typically contain a grid with these columns:
  காலை / Morning | மதியம் / Afternoon | இரவு / Night
  (Hindi: सुबह | दोपहर | रात)
Each cell contains a number: 1 = take, 0 = skip, ½ = half dose.

## STEP 2 — Extract Each Medicine Row
For EACH medicine, extract:
  • name: The medicine/drug name exactly as written (correct obvious misspellings if confident).
  • dosage: The strength per intake (e.g., "500mg", "250mg/5ml", "1 Tablet"). 
    ⚠️ This is NOT the frequency number from the grid. If no strength is written, use "Not specified".
  • quantity: Total count dispensed (e.g., "10 tablets", "1 strip"). Use null if not mentioned.
  • period_of_intake: Combine the grid numbers into "M-A-N" format AND translate to English.
    Examples:
      1 under Morning, 0 under Afternoon, 1 under Night → "1-0-1 (Morning, Night)"
      1 under Morning, 1 under Afternoon, 1 under Night → "1-1-1 (Morning, Afternoon, Night)"
      0 under Morning, 0 under Afternoon, 1 under Night → "0-0-1 (Night only)"
      ½ under Morning, 0 under Afternoon, ½ under Night → "½-0-½ (Morning half, Night half)"
    If no grid exists, describe the frequency in words (e.g., "Twice daily", "Once at night").
  • route: Oral, Topical, IV, IM, Inhaled, etc. Use null if not mentioned.
  • duration: "5 days", "1 week", etc. Use null if not mentioned.
  • instructions: "After food", "Before sleep", "Empty stomach", etc. Use null if not mentioned.

## STEP 3 — Doctor Notes
Extract any diagnosis, advice, or follow-up notes (e.g., "Review after 5 days", "Drink warm water").

## FEW-SHOT EXAMPLE
If the prescription image shows:
  Tab. Amoxicillin 500mg    1  0  1   x 5 days   (After food)
  Syp. Paracetamol 250mg    1  1  1   x 3 days
  Cap. Omeprazole 20mg      1  0  0   x 7 days   (Before food)

Then your output MUST be:
{
  "medicines": [
    {"name": "Tab. Amoxicillin", "dosage": "500mg", "quantity": null, "period_of_intake": "1-0-1 (Morning, Night)", "route": "Oral", "duration": "5 days", "instructions": "After food"},
    {"name": "Syp. Paracetamol", "dosage": "250mg", "quantity": null, "period_of_intake": "1-1-1 (Morning, Afternoon, Night)", "route": "Oral", "duration": "3 days", "instructions": null},
    {"name": "Cap. Omeprazole", "dosage": "20mg", "quantity": null, "period_of_intake": "1-0-0 (Morning only)", "route": "Oral", "duration": "7 days", "instructions": "Before food"}
  ],
  "doctor_notes": null
}

Now analyse the attached image and return ONLY a valid JSON object following the exact schema above."""

# ──────────────────────────────────────────────────────────────────────────────
# JSON Extraction Helper
# ──────────────────────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> str:
    """
    Robustly extract a JSON object from a VLM response that may contain
    markdown code fences, preamble text, or trailing explanation.
    """
    # Strategy 1: Find ```json ... ``` block
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if md_match:
        return md_match.group(1).strip()

    # Strategy 2: Find the first { ... last }
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return raw[first_brace : last_brace + 1].strip()

    # Fallback: return as-is and let Pydantic raise a clear error
    return raw.strip()

# ──────────────────────────────────────────────────────────────────────────────
# VLM Extractor
# ──────────────────────────────────────────────────────────────────────────────

MAX_RETRIES = 2

def extract_prescription(image_bytes: bytes, mime_type: str = "image/jpeg") -> PrescriptionAnalysis:
    """
    Sends the raw image to a local VLM (via Ollama) and returns a structured Pydantic object.
    Uses a two-pass system+user prompt with retry logic and temperature escalation.
    """
    model_name = os.environ.get("VLM_MODEL", "qwen2.5vl")
    logger.info(f"Calling Local VLM (Ollama/{model_name})...")

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        temperature = 0.1 * attempt  # 0.1 → 0.2 on retry (slightly more creative)
        logger.debug(f"Attempt {attempt}/{MAX_RETRIES} (temperature={temperature})")

        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": USER_PROMPT,
                        "images": [image_bytes],
                    },
                ],
                options={
                    "temperature": temperature,
                    "num_predict": 4096,
                },
            )

            content = response["message"]["content"]
            logger.debug(f"Raw VLM response ({len(content)} chars): {content[:300]}...")

            clean_json = _extract_json(content)
            result = PrescriptionAnalysis.model_validate_json(clean_json)

            logger.info(
                f"✅ VLM extracted {len(result.medicines)} medicine(s) on attempt {attempt}"
            )
            return result

        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt} failed: {e}")

    # All retries exhausted
    logger.exception(f"Local VLM Extraction failed after {MAX_RETRIES} attempts: {last_error}")
    raise last_error  # type: ignore[misc]
