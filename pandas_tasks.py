"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    df = pd.read_csv(data.path)
    missing = df.isna().sum()
    older_than_30 = df[df["Age"] > 30].shape[0]
    mean_age_by_class = df.groupby("Pclass")["Age"].mean()
    survival_by_class = df.groupby("Pclass")["Survived"].mean()
    top5_fare = df["Fare"].sort_values(ascending=False).head(5).tolist()

    return TitanicSummary(
        missing=missing,
        older_than_30=older_than_30,
        mean_age_by_class=mean_age_by_class,
        survival_by_class=survival_by_class,
        top5_fare=top5_fare,
    )
