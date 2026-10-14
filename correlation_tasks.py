"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import (
    BrainCorrelationSummary,
    BrainDataInput,
)


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    df = pd.read_csv(data.path, sep="\t", na_values=["NA"])
    males = df[df["Sex"] == "Male"]
    females = df[df["Sex"] == "Female"]
    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
    male_corr = {}
    for f in features:
        male_corr[f] = males[f].corr(males["MRI_Count"])
    female_corr = {}
    for f in features:
        female_corr[f] = females[f].corr(females["MRI_Count"])
    
    strongest = None
    max_abs_corr = -1  
    for f in features:
        if abs(male_corr[f]) > max_abs_corr:
            max_abs_corr = abs(male_corr[f])
            strongest = f
        if abs(female_corr[f]) > max_abs_corr:
            max_abs_corr = abs(female_corr[f])
            strongest = f

    return BrainCorrelationSummary(
        male_correlations=male_corr,
        female_correlations=female_corr,
        strongest_mri_feature=strongest,
    )
