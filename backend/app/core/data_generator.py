"""
AI Panchayat – Synthetic Dataset Generator
Generates 5 exhaustive, realistic DataFrames (1000+ rows each)
with complex mathematical relationships and hidden biases.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Corporate Hiring – Bias: penalizes females despite high tech scores
# ---------------------------------------------------------------------------
def _generate_corporate_hiring(n: int = 1200) -> pd.DataFrame:
    gender = np.random.choice(["Male", "Female"], size=n, p=[0.55, 0.45])
    years_exp = np.random.poisson(lam=5, size=n).clip(0, 25)
    tech_score = np.random.normal(72, 12, size=n).clip(0, 100).round(1)
    culture_fit = np.random.normal(65, 15, size=n).clip(0, 100).round(1)
    education = np.random.choice(
        ["High School", "Bachelors", "Masters", "PhD"],
        size=n, p=[0.10, 0.45, 0.35, 0.10],
    )
    edu_map = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
    edu_num = np.array([edu_map[e] for e in education])

    # Hidden bias: females get a penalty of -15 on their latent score
    latent = (
        0.30 * tech_score
        + 0.25 * culture_fit
        + 3.0 * years_exp
        + 5.0 * edu_num
        + np.random.normal(0, 8, size=n)
    )
    gender_penalty = np.where(np.array(gender) == "Female", -15, 0)
    latent += gender_penalty

    hired = (latent > np.percentile(latent, 60)).astype(int)

    return pd.DataFrame({
        "gender": gender,
        "years_experience": years_exp,
        "tech_score": tech_score,
        "culture_fit_score": culture_fit,
        "education_level": education,
        "hired": hired,
    })


# ---------------------------------------------------------------------------
# 2. Mortgage Approvals – Bias: penalizes specific zip codes (race proxy)
# ---------------------------------------------------------------------------
def _generate_mortgage_approvals(n: int = 1100) -> pd.DataFrame:
    race = np.random.choice(
        ["White", "Black", "Hispanic", "Asian"],
        size=n, p=[0.55, 0.18, 0.17, 0.10],
    )
    # Zip codes correlated with race (proxy discrimination)
    zip_map = {"White": "300xx", "Black": "100xx", "Hispanic": "200xx", "Asian": "400xx"}
    zip_code = np.array([zip_map[r] for r in race])
    # Add some noise – 20 % random reassignment
    noise_mask = np.random.rand(n) < 0.20
    zip_code[noise_mask] = np.random.choice(list(zip_map.values()), size=noise_mask.sum())

    income = np.random.lognormal(mean=11.0, sigma=0.5, size=n).round(0)
    credit_score = np.random.normal(680, 60, size=n).clip(300, 850).round(0).astype(int)
    loan_amount = (income * np.random.uniform(2.5, 5.0, size=n)).round(0)
    dti_ratio = np.random.uniform(0.15, 0.55, size=n).round(3)

    latent = (
        0.004 * credit_score
        + 0.00001 * income
        - 1.5 * dti_ratio
        + np.random.normal(0, 0.4, size=n)
    )
    # Zip-code penalty — proxy for racial bias
    zip_penalty = np.where(zip_code == "100xx", -0.8, np.where(zip_code == "200xx", -0.5, 0.0))
    latent += zip_penalty

    approved = (latent > np.median(latent)).astype(int)

    return pd.DataFrame({
        "race": race,
        "zip_code": zip_code,
        "annual_income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "dti_ratio": dti_ratio,
        "approved": approved,
    })


# ---------------------------------------------------------------------------
# 3. Hospital Triage – Bias: under-prioritizes uninsured for ICU beds
# ---------------------------------------------------------------------------
def _generate_hospital_triage(n: int = 1050) -> pd.DataFrame:
    insurance = np.random.choice(
        ["Private", "Medicare", "Medicaid", "Uninsured"],
        size=n, p=[0.40, 0.25, 0.20, 0.15],
    )
    age = np.random.normal(55, 18, size=n).clip(18, 95).round(0).astype(int)
    severity_score = np.random.normal(50, 20, size=n).clip(0, 100).round(1)
    comorbidities = np.random.poisson(lam=2, size=n).clip(0, 8)
    vitals_index = np.random.normal(60, 15, size=n).clip(0, 100).round(1)

    latent = (
        0.40 * severity_score
        + 0.25 * vitals_index
        + 3.0 * comorbidities
        + 0.1 * age
        + np.random.normal(0, 6, size=n)
    )
    # Bias: uninsured patients get demoted
    ins_penalty = np.where(np.array(insurance) == "Uninsured", -18, 0)
    latent += ins_penalty

    icu_admitted = (latent > np.percentile(latent, 55)).astype(int)

    return pd.DataFrame({
        "insurance_status": insurance,
        "age": age,
        "severity_score": severity_score,
        "comorbidities": comorbidities,
        "vitals_index": vitals_index,
        "icu_admitted": icu_admitted,
    })


# ---------------------------------------------------------------------------
# 4. Criminal Recidivism – Bias: over-predicts risk for minorities
# ---------------------------------------------------------------------------
def _generate_criminal_recidivism(n: int = 1150) -> pd.DataFrame:
    race = np.random.choice(
        ["White", "Black", "Hispanic", "Other"],
        size=n, p=[0.50, 0.25, 0.18, 0.07],
    )
    age = np.random.normal(32, 10, size=n).clip(18, 70).round(0).astype(int)
    prior_offenses = np.random.poisson(lam=1.5, size=n).clip(0, 12)
    employment = np.random.choice(["Employed", "Unemployed"], size=n, p=[0.6, 0.4])
    substance_abuse = np.random.choice([0, 1], size=n, p=[0.65, 0.35])
    social_support = np.random.normal(50, 15, size=n).clip(0, 100).round(1)

    latent = (
        5.0 * prior_offenses
        - 0.15 * age
        + 8.0 * substance_abuse
        - 0.10 * social_support
        + 6.0 * (np.array(employment) == "Unemployed").astype(float)
        + np.random.normal(0, 5, size=n)
    )
    # Bias: inflate risk for Black & Hispanic
    race_arr = np.array(race)
    race_penalty = np.where(race_arr == "Black", 10, np.where(race_arr == "Hispanic", 6, 0))
    latent += race_penalty

    recidivism = (latent > np.percentile(latent, 50)).astype(int)

    return pd.DataFrame({
        "race": race,
        "age": age,
        "prior_offenses": prior_offenses,
        "employment_status": employment,
        "substance_abuse_history": substance_abuse,
        "social_support_score": social_support,
        "recidivism": recidivism,
    })


# ---------------------------------------------------------------------------
# 5. University Admissions – Bias: favors legacy students
# ---------------------------------------------------------------------------
def _generate_university_admissions(n: int = 1100) -> pd.DataFrame:
    legacy = np.random.choice([0, 1], size=n, p=[0.80, 0.20])
    gpa = np.random.normal(3.3, 0.45, size=n).clip(1.5, 4.0).round(2)
    sat_score = np.random.normal(1200, 150, size=n).clip(600, 1600).round(0).astype(int)
    extracurriculars = np.random.poisson(lam=3, size=n).clip(0, 10)
    essay_score = np.random.normal(70, 15, size=n).clip(0, 100).round(1)
    household_income = np.random.lognormal(mean=11.2, sigma=0.6, size=n).round(0)

    latent = (
        12.0 * gpa
        + 0.015 * sat_score
        + 2.0 * extracurriculars
        + 0.20 * essay_score
        + np.random.normal(0, 4, size=n)
    )
    # Bias: legacy students get a massive boost
    legacy_boost = np.where(legacy == 1, 14, 0)
    latent += legacy_boost

    admitted = (latent > np.percentile(latent, 55)).astype(int)

    return pd.DataFrame({
        "legacy_status": legacy,
        "gpa": gpa,
        "sat_score": sat_score,
        "extracurriculars": extracurriculars,
        "essay_score": essay_score,
        "household_income": household_income,
        "admitted": admitted,
    })


# ---------------------------------------------------------------------------
# Registry & public API
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, dict] = {
    "corporate_hiring": {
        "generator": _generate_corporate_hiring,
        "target": "hired",
        "sensitive": "gender",
        "label": "Corporate Hiring",
    },
    "mortgage_approvals": {
        "generator": _generate_mortgage_approvals,
        "target": "approved",
        "sensitive": "race",
        "label": "Mortgage Approvals",
    },
    "hospital_triage": {
        "generator": _generate_hospital_triage,
        "target": "icu_admitted",
        "sensitive": "insurance_status",
        "label": "Hospital Triage",
    },
    "criminal_recidivism": {
        "generator": _generate_criminal_recidivism,
        "target": "recidivism",
        "sensitive": "race",
        "label": "Criminal Recidivism",
    },
    "university_admissions": {
        "generator": _generate_university_admissions,
        "target": "admitted",
        "sensitive": "legacy_status",
        "label": "University Admissions",
    },
}

AVAILABLE_DATASETS = list(_REGISTRY.keys())


def get_test_dataset(name: str) -> Tuple[pd.DataFrame, str, str]:
    """Return (DataFrame, target_column, sensitive_column) for the named dataset."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {AVAILABLE_DATASETS}")
    entry = _REGISTRY[name]
    df = entry["generator"]()
    return df, entry["target"], entry["sensitive"]
