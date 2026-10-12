"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import (
    BrainCorrelationSummary,
    BrainDataInput,
)


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    df = pd.read_csv(data.path, sep="\t")
    df = df.fillna(df.mean(numeric_only=True))
    males = df[df["Sex"] == "Male"]
    females = df[df["Sex"] == "Female"]
    features = ["FSIO", "VIQ", "PIQ", "Weight", "Height"]
    male_corr = {}
    for f in features:
        male_corr[f] = males[f].corr(males["MRI_Count"])
    female_corr = {}
    for f in features:
        female_corr[f] = females[f].corr(females["MRI_Count"])
    all_corr = {}
    for f in features:
        all_corr[f] = (male_corr[f] + female_corr[f]) / 2

    strongest = None
    max_corr = -2  
    for f in features:
        if all_corr[f] > max_corr:
            max_corr = all_corr[f]
            strongest = f

    return BrainCorrelationSummary(
        male_correlations=male_corr,
        female_correlations=female_corr,
        strongest_mri_feature=strongest,
    )
