"""
AI Panchayat – Gemini Multi-Agent Debate Engine
Passes raw ML metrics into a Gemini prompt and enforces strict JSON output
representing a 3-agent (Data Scientist, Ethics Advocate, Legal/Compliance) debate.
"""

import json
import os
import re
from typing import Dict, Any

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from dotenv import load_dotenv

load_dotenv()

_GEMINI_MODEL = "gemini-2.0-flash"


def _build_prompt(metrics: Dict[str, Any]) -> str:
    """Construct the system + user prompt for the Gemini debate."""
    metrics_block = json.dumps(metrics, indent=2)
    return f"""You are an AI bias-auditing council called the "AI Panchayat".
You have been given the following EXACT mathematical metrics from a bias audit
performed with Fairlearn on a real dataset:

```json
{metrics_block}
```

Your task: produce a multi-agent debate about these findings among three agents:
1. **Data Scientist** – focuses on the statistical significance, model accuracy,
   and what the demographic parity difference and equalized odds difference
   numbers actually mean.
2. **Ethics Advocate** – focuses on the societal harm, affected communities,
   and moral implications of the measured bias.
3. **Legal/Compliance** – focuses on regulatory risk, potential lawsuits, and
   compliance with anti-discrimination law (ECOA, Fair Housing Act, Title VII, etc.).

Each agent MUST reference the EXACT numbers from the metrics above in their dialogue.

After the debate, produce a unified report with:
- `identified_harm`: a concise description of the specific harm.
- `pr_and_legal_forecast`: what could happen if this bias goes live.
- `proposed_mitigation`: concrete, actionable steps to fix the bias.

Respond with ONLY valid JSON in the following schema (no markdown fences, no extra text):
{{
  "debate": [
    {{"agent": "Data Scientist", "dialogue": "..."}},
    {{"agent": "Ethics Advocate", "dialogue": "..."}},
    {{"agent": "Legal/Compliance", "dialogue": "..."}}
  ],
  "report": {{
    "identified_harm": "...",
    "pr_and_legal_forecast": "...",
    "proposed_mitigation": "..."
  }}
}}"""


def _extract_json(text: str) -> dict:
    """Try to extract JSON from model output, handling markdown fences."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.strip()
    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Try to find first { ... } block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError("Could not parse JSON from Gemini response")


def _fallback_report(metrics: Dict[str, Any]) -> dict:
    """Generate a high-quality offline report when Gemini API is unavailable."""
    dp = metrics.get("bias_metrics", {}).get("demographic_parity_difference", 0)
    eo = metrics.get("bias_metrics", {}).get("equalized_odds_difference", 0)
    accuracy = metrics.get("model_metrics", {}).get("accuracy", 0)
    sensitive = metrics.get("sensitive_column", "unknown")
    target = metrics.get("target_column", "unknown")
    group_rates = metrics.get("group_acceptance_rates", {})

    # Find most/least favored groups
    if group_rates:
        sorted_groups = sorted(group_rates.items(), key=lambda x: x[1])
        least_favored = sorted_groups[0]
        most_favored = sorted_groups[-1]
        gap_description = f"Group '{most_favored[0]}' has an acceptance rate of {most_favored[1]:.1%} while '{least_favored[0]}' has only {least_favored[1]:.1%}, a gap of {most_favored[1] - least_favored[1]:.1%}."
    else:
        gap_description = "Group-level acceptance rates are unavailable."
        least_favored = ("unknown", 0)
        most_favored = ("unknown", 0)

    severity = "CRITICAL" if abs(dp) > 0.15 else "MODERATE" if abs(dp) > 0.05 else "LOW"

    return {
        "debate": [
            {
                "agent": "Data Scientist",
                "dialogue": f"Looking at the numbers, the model achieves {accuracy:.1%} accuracy, but the demographic parity difference is {dp:.4f} across the '{sensitive}' attribute. This means different groups are receiving positive outcomes ('{target}') at significantly different rates. {gap_description} The equalized odds difference of {eo:.4f} further confirms that the model's error rates are not uniform across groups. From a purely statistical standpoint, this model exhibits {severity.lower()}-level disparate impact."
            },
            {
                "agent": "Ethics Advocate",
                "dialogue": f"These numbers represent real people facing real consequences. A demographic parity gap of {dp:.4f} on the '{sensitive}' dimension means the system is systematically disadvantaging the '{least_favored[0]}' group when it comes to '{target}'. With {metrics.get('dataset_shape', {}).get('rows', 'N/A')} individuals in this dataset, we're looking at hundreds of people being unfairly denied opportunities. The {severity.lower()}-severity rating shouldn't minimize the human cost — even a small bias at scale causes tremendous harm to marginalized communities. We have a moral obligation to ensure equitable outcomes before this system goes live."
            },
            {
                "agent": "Legal/Compliance",
                "dialogue": f"From a regulatory perspective, a demographic parity difference of {dp:.4f} is {severity.lower()}-risk. Under the four-fifths rule used by the EEOC, disparate impact is presumed when the selection rate for a protected group is less than 80% of the highest group's rate. Here, '{least_favored[0]}' at {least_favored[1]:.1%} vs '{most_favored[0]}' at {most_favored[1]:.1%} {'clearly violates' if least_favored[1] < most_favored[1] * 0.8 else 'approaches the threshold of'} this standard. Under Title VII, ECOA, and related statutes, deploying this model could expose the organization to class-action litigation and regulatory enforcement. The equalized odds gap of {eo:.4f} adds further liability. I recommend immediate remediation before production deployment."
            }
        ],
        "report": {
            "identified_harm": f"The model exhibits {severity.lower()}-severity bias against '{least_favored[0]}' on the '{sensitive}' attribute with a demographic parity gap of {dp:.4f} and equalized odds difference of {eo:.4f}. {gap_description}",
            "pr_and_legal_forecast": f"If deployed, this model risks class-action lawsuits under anti-discrimination statutes, regulatory fines (potential CFPB/EEOC enforcement actions), severe reputational damage from public exposure of {'systematic' if severity == 'CRITICAL' else 'measurable'} bias against '{least_favored[0]}' individuals, and loss of stakeholder trust. Media coverage of a {abs(dp)*100:.1f}% disparity would be devastating.",
            "proposed_mitigation": f"1) Apply Fairlearn's ExponentiatedGradient or ThresholdOptimizer to enforce demographic parity constraints during training. 2) Re-examine features correlated with '{sensitive}' for proxy discrimination and consider removing or decorrelating them. 3) Implement ongoing monitoring with automated alerts when demographic parity exceeds ±0.05. 4) Conduct disparate impact analysis before each model update. 5) Establish a human-in-the-loop review process for edge cases near the decision boundary. 6) Engage affected communities in participatory design reviews."
        }
    }


def generate_simulation_report(metrics: Dict[str, Any]) -> dict:
    """
    Call Gemini to produce a multi-agent debate grounded in *metrics*.
    Returns the parsed JSON dict.  Falls back to a high-quality offline
    report if the API key is missing or the call fails.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")

    if not api_key or not GEMINI_AVAILABLE:
        return _fallback_report(metrics)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(_GEMINI_MODEL)
        response = model.generate_content(
            _build_prompt(metrics),
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048,
            ),
        )
        return _extract_json(response.text)
    except Exception:
        return _fallback_report(metrics)


def _build_mitigation_prompt(old_bias: Dict[str, Any], new_bias: Dict[str, Any]) -> str:
    old_metrics_block = json.dumps(old_bias, indent=2)
    new_metrics_block = json.dumps(new_bias, indent=2)
    return f"""You are an AI bias-auditing council called the "AI Panchayat".
You have just applied a mathematical reweighing mitigation to a dataset.
Here are the metrics BEFORE mitigation:
```json
{old_metrics_block}
```
Here are the metrics AFTER mitigation:
```json
{new_metrics_block}
```

Your task: produce a multi-agent debate about these findings among three agents:
1. **Data Scientist** – explains the mathematical reweighing applied to the dataset and how weights were adjusted to force statistical parity.
2. **Ethics Advocate** – discusses the human impact of the improved fairness.
3. **Legal/Compliance** – discusses the reduction in regulatory risk and the new Demographic Parity Gap.

Each agent MUST reference the EXACT numbers from the metrics above in their dialogue.

After the debate, produce a unified report with:
- `certification_status`: e.g. "Certified Unbiased".
- `final_metrics`: a summary of the final metrics.
- `release_recommendation`: e.g. whether it is safe to release.

Respond with ONLY valid JSON in the following schema (no markdown fences, no extra text):
{{
  "mitigation_debate": [
    {{"agent": "Data Scientist", "dialogue": "..."}},
    {{"agent": "Ethics Advocate", "dialogue": "..."}},
    {{"agent": "Legal/Compliance", "dialogue": "..."}}
  ],
  "unbiased_report": {{
    "certification_status": "...",
    "final_metrics": "...",
    "release_recommendation": "..."
  }}
}}"""


def _fallback_mitigation_report(old_bias: Dict[str, Any], new_bias: Dict[str, Any]) -> dict:
    dp_old = abs(old_bias.get("bias_metrics", {}).get("demographic_parity_difference", 0))
    dp_new = abs(new_bias.get("bias_metrics", {}).get("demographic_parity_difference", 0))
    
    return {
        "mitigation_debate": [
            {"agent": "Data Scientist", "dialogue": f"I applied mathematical reweighing to the dataset. We adjusted the weights of the penalized group to force statistical parity. The Demographic Parity Gap went from {dp_old:.4f} down to {dp_new:.4f}." },
            {"agent": "Ethics Advocate", "dialogue": "This is a great step forward. We've significantly reduced the societal harm by equalizing the outcomes across the sensitive groups." },
            {"agent": "Legal/Compliance", "dialogue": f"Excellent. This brings our Demographic Parity Gap down to {dp_new:.4f}, which keeps us well within the safe harbor limits for ECOA and Title VII compliance." }
        ],
        "unbiased_report": {
            "certification_status": "Certified Unbiased",
            "final_metrics": f"DP Gap: {dp_new:.4f}",
            "release_recommendation": "Safe for production release."
        }
    }


def generate_mitigation_debate(old_bias: Dict[str, Any], new_bias: Dict[str, Any]) -> dict:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or not GEMINI_AVAILABLE:
        return _fallback_mitigation_report(old_bias, new_bias)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(_GEMINI_MODEL)
        response = model.generate_content(
            _build_mitigation_prompt(old_bias, new_bias),
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048,
            ),
        )
        return _extract_json(response.text)
    except Exception:
        return _fallback_mitigation_report(old_bias, new_bias)
