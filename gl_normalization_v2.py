
"""
GL Normalization Assistant — Version 2

Screens General Ledger data for transactions that deserve analyst review
as potentially non-recurring expenses.

This is a screening model, not an automatic accounting/valuation decision.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


DEFAULT_TOP_N = 100
MIN_ABS_AMOUNT = 100.0
REVIEW_QUEUE_THRESHOLD = 0.25
REVIEW_PRIORITY_THRESHOLDS = (REVIEW_QUEUE_THRESHOLD, 0.50, 0.75)

# Starting weights only. These are NOT claimed to be professionally validated.
WEIGHTS = {
    "amount_anomaly": 0.28,
    "historical_recurrence": 0.32,
    "semantic_novelty": 0.18,
    "vendor_novelty": 0.07,
    "frequency_rarity": 0.05,
    "year_end_context": 0.03,
    "journal_anomaly": 0.07,
}

EXPENSE_KEYWORDS = {
    "expense", "expenses", "cost", "costs", "rent", "salary", "salaries",
    "wage", "wages", "legal", "consulting", "professional", "insurance",
    "utilities", "marketing", "advertising", "repairs", "maintenance",
    "depreciation", "impairment", "restructuring", "litigation", "audit",
    "travel", "freight", "logistics", "commission", "office", "training",
    "interest", "bank fee", "bank fees", "software", "subscription",
    "technology", "research", "development"
}


def _normalise_columns(columns: Iterable[str]) -> dict[str, str]:
    return {
        c: re.sub(
            r"_+",
            "_",
            re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower())
        ).strip("_")
        for c in columns
    }


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def clean_text(text: object) -> str:
    text = "" if pd.isna(text) else str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    stop_terms = {
        "invoice", "inv", "payment", "pay", "journal", "jv",
        "entry", "transaction", "txn", "doc", "document",
        "reference", "ref"
    }
    return " ".join(t for t in text.split() if t not in stop_terms)


def safe_minmax(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lo, hi = x.min(), x.max()
    if hi <= lo:
        return pd.Series(np.zeros(len(x)), index=series.index)
    return (x - lo) / (hi - lo)


def robust_zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    valid = x.dropna()

    if valid.empty:
        return pd.Series(np.zeros(len(x)), index=series.index)

    med = valid.median()
    mad = np.median(np.abs(valid - med))

    if mad == 0 or pd.isna(mad):
        std = valid.std()
        if std == 0 or pd.isna(std):
            return pd.Series(np.zeros(len(x)), index=series.index)
        return (x - valid.mean()) / std

    return 0.6745 * (x - med) / mad


def token_jaccard(a: str, b: str) -> float:
    A, B = set(a.split()), set(b.split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def load_gl(path: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(path, pd.DataFrame):
        df = path.copy()
    else:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            raise ValueError("Please upload CSV, XLSX or XLS.")

    source_row_count = len(df)
    df = df.rename(columns=_normalise_columns(df.columns))

    aliases = {
        "date": ["date", "posting_date", "transaction_date", "journal_date"],
        "account": ["account", "account_name", "gl_account", "account_code"],
        "sub_account": ["sub_account", "subaccount", "sub_account_name"],
        "description": [
            "description", "memo", "narrative", "details",
            "transaction_description"
        ],
        "amount": ["amount", "value", "transaction_amount"],
        "debit": ["debit", "debits"],
        "credit": ["credit", "credits"],
        "journal_id": [
            "journal_id", "journal", "journal_number",
            "document_number", "doc_number"
        ],
        "vendor": ["vendor", "supplier", "payee", "counterparty"],
        "cost_center": ["cost_center", "cost_centre", "department"],
        "account_type": [
            "account_type", "account_category", "account_class"
        ],
    }

    selected = {}
    for target, choices in aliases.items():
        col = _first_existing(df, choices)
        if col:
            selected[target] = col

    missing = [x for x in ["date", "account", "description"] if x not in selected]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + ". Required: date, account, description, and amount "
              "OR debit/credit."
        )

    if "amount" not in selected and not any(
        column in selected for column in ["debit", "credit"]
    ):
        raise ValueError(
            "Missing amount data. Include an amount column or a debit/credit column."
        )

    df = df.rename(columns={v: k for k, v in selected.items()}).copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["account"] = df["account"].fillna("").astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()

    if "amount" not in df.columns:
        debit = pd.to_numeric(
            df["debit"]
            if "debit" in df.columns
            else pd.Series(0, index=df.index),
            errors="coerce"
        ).fillna(0)
        credit = pd.to_numeric(
            df["credit"]
            if "credit" in df.columns
            else pd.Series(0, index=df.index),
            errors="coerce"
        ).fillna(0)
        df["amount"] = debit - credit
    else:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    for optional in [
        "sub_account", "vendor", "cost_center", "account_type"
    ]:
        if optional not in df.columns:
            df[optional] = ""

    # Use the real journal/document ID when available.
    # Otherwise create a weaker synthetic grouping and mark it as such.
    if "journal_id" not in df.columns:
        df["journal_id"] = (
            df["date"].dt.strftime("%Y%m%d").fillna("unknown")
            + "_"
            + df["description"].map(clean_text)
        )
        df["journal_id_is_synthetic"] = True
    else:
        df["journal_id"] = df["journal_id"].fillna("").astype(str).str.strip()
        missing_journal = df["journal_id"].eq("")
        df.loc[missing_journal, "journal_id"] = (
            df.loc[missing_journal, "date"].dt.strftime("%Y%m%d")
            .fillna("unknown")
            + "_"
            + df.loc[missing_journal, "description"].map(clean_text)
        )
        df["journal_id_is_synthetic"] = missing_journal

    valid_rows = (
        df["date"].notna()
        & df["amount"].notna()
        & df["description"].ne("")
        & df["account"].ne("")
    )
    dropped_row_count = int((~valid_rows).sum())
    df = df.loc[valid_rows].copy()

    if df.empty:
        raise ValueError(
            "No usable transactions remain after cleaning "
            f"({dropped_row_count:,} of {source_row_count:,} rows excluded)."
        )

    df["year"] = df["date"].dt.year.astype(int)
    df["abs_amount"] = df["amount"].abs()
    df["clean_description"] = df["description"].map(clean_text)

    df = df.reset_index(drop=True)
    df.attrs["source_row_count"] = source_row_count
    df.attrs["dropped_row_count"] = dropped_row_count
    return df


def classify_expense_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    account_type = df["account_type"].fillna("").astype(str).str.lower()
    known_expense_type = account_type.str.contains(
        r"expense|cost|opex|operating", regex=True
    )
    known_non_expense_type = account_type.str.contains(
        r"asset|liabilit|equity|revenue|income|capital|balance[\s_-]*sheet",
        regex=True,
    )

    account_text = (
        df["account"].fillna("").astype(str)
        + " "
        + df["sub_account"].fillna("").astype(str)
    ).str.lower()

    keyword_pattern = "|".join(
        re.escape(x) for x in sorted(
            EXPENSE_KEYWORDS, key=len, reverse=True
        )
    )

    heuristic_expense = account_text.str.contains(
        keyword_pattern, regex=True, na=False
    )

    unclassified_type = ~(known_expense_type | known_non_expense_type)
    df["is_expense_candidate"] = (
        (known_expense_type & ~known_non_expense_type)
        | (unclassified_type & heuristic_expense)
    )
    return df


def add_journal_features(df: pd.DataFrame) -> pd.DataFrame:
    journal = (
        df.groupby("journal_id")
        .agg(
            journal_line_count=("journal_id", "size"),
            journal_total_abs=("abs_amount", "sum"),
            journal_net=("amount", "sum"),
            journal_account_count=("account", "nunique"),
        )
        .reset_index()
    )

    out = df.merge(journal, on="journal_id", how="left")

    out["journal_size_z"] = (
        robust_zscore(out["journal_total_abs"])
        .clip(lower=0, upper=8)
    )
    out["journal_anomaly"] = out["journal_size_z"] / 8.0

    return out


def add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["description_count"] = (
        out.groupby(["account", "clean_description"])["journal_id"]
        .transform("nunique")
    )

    out["vendor"] = out["vendor"].fillna("").astype(str).str.strip()
    has_vendor = out["vendor"].ne("")
    out["vendor_count"] = 0
    if has_vendor.any():
        out.loc[has_vendor, "vendor_count"] = (
            out.loc[has_vendor]
            .groupby(["account", "vendor"])["journal_id"]
            .transform("nunique")
        )

    out["account_journal_count"] = (
        out.groupby("account")["journal_id"]
        .transform("nunique")
    )

    out["frequency_rarity"] = (
        1.0 / np.sqrt(out["description_count"].clip(lower=1))
    )
    out["frequency_rarity"] = safe_minmax(out["frequency_rarity"])

    return out


def add_amount_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["account_amount_z"] = (
        out.groupby("account")["abs_amount"]
        .transform(robust_zscore)
        .fillna(0)
    )

    out["amount_anomaly"] = (
        out["account_amount_z"].clip(lower=0, upper=8) / 8.0
    )

    total_expense = out.loc[
        out["is_expense_candidate"], "abs_amount"
    ].sum()

    if total_expense > 0:
        out["percent_of_expense_population"] = (
            out["abs_amount"] / total_expense
        )
    else:
        out["percent_of_expense_population"] = 0.0

    return out


def add_year_end_context(
    df: pd.DataFrame,
    fiscal_year_end_month: int = 12,
) -> pd.DataFrame:
    out = df.copy()

    if not 1 <= fiscal_year_end_month <= 12:
        raise ValueError("Fiscal year-end month must be between 1 and 12.")

    dates = pd.to_datetime(out["date"])
    fiscal_year_end_year = dates.dt.year + (
        dates.dt.month > fiscal_year_end_month
    ).astype(int)
    fiscal_year_end = pd.to_datetime(
        {
            "year": fiscal_year_end_year,
            "month": fiscal_year_end_month,
            "day": 1,
        }
    ) + pd.offsets.MonthEnd(0)

    out["days_from_year_end"] = (
        fiscal_year_end - dates
    ).dt.days.clip(lower=0)

    # Small contextual signal only; it is not evidence by itself.
    out["year_end_context"] = np.where(
        out["days_from_year_end"] <= 30, 1.0, 0.0
    )

    return out


def add_historical_recurrence(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each transaction, search prior years in the same account.

    A prior transaction is treated as similar when:
    - amount is within +/-40%, AND
    - description has token overlap OR vendor is the same.

    This is a transparent MVP recurrence model, not a finished semantic model.
    """

    out = df.copy()

    history = {}
    for idx, row in out.iterrows():
        key = (row["account"], int(row["year"]))
        history.setdefault(key, []).append(idx)

    rows = out.to_dict("index")
    min_year = int(out["year"].min())

    similar_count = []
    years_seen_count = []

    for idx, row in out.iterrows():
        account = row["account"]
        year = int(row["year"])
        amount = abs(float(row["amount"]))
        text = row["clean_description"]
        vendor = str(row.get("vendor", "")).strip().lower()

        count = 0
        years_seen = set()

        for prior_year in range(year - 1, min_year - 1, -1):
            for old_idx in history.get((account, prior_year), []):
                old = rows[old_idx]
                old_amount = abs(float(old["amount"]))

                if amount == 0:
                    continue

                if not (0.60 * amount <= old_amount <= 1.40 * amount):
                    continue

                text_sim = token_jaccard(
                    text, old["clean_description"]
                )

                old_vendor = str(
                    old.get("vendor", "")
                ).strip().lower()

                vendor_match = (
                    vendor != ""
                    and old_vendor != ""
                    and vendor == old_vendor
                )

                if text_sim >= 0.20 or vendor_match:
                    count += 1
                    years_seen.add(prior_year)

        similar_count.append(count)
        years_seen_count.append(len(years_seen))

    out["historical_similar_count"] = similar_count
    out["historical_years_with_similar_activity"] = years_seen_count

    out["historical_recurrence_strength"] = np.minimum(
        out["historical_years_with_similar_activity"] / 3.0,
        1.0
    )

    # High = weak historical recurrence = more reason for review.
    out["historical_recurrence_anomaly"] = (
        1.0 - out["historical_recurrence_strength"]
    )

    return out


def add_semantic_novelty(df: pd.DataFrame) -> pd.DataFrame:
    """
    TF-IDF baseline for description novelty.

    This is deliberately the baseline NLP method so the team can later
    compare it against sentence embeddings.
    """

    out = df.copy()
    text = out["clean_description"].fillna("")

    has_text = text.str.strip().ne("")
    out["semantic_novelty"] = 0.0
    if not has_text.any():
        return out

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        min_df=1,
        max_features=10000,
    )

    X = vectorizer.fit_transform(text.loc[has_text])
    centroid = np.asarray(X.mean(axis=0))

    similarity = cosine_similarity(X, centroid).ravel()
    novelty = 1.0 - similarity

    out.loc[has_text, "semantic_novelty"] = safe_minmax(
        pd.Series(novelty, index=text.index[has_text])
    )

    return out


def add_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    features = out[
        [
            "abs_amount",
            "description_count",
            "account_journal_count",
            "days_from_year_end",
            "journal_total_abs",
            "journal_line_count",
        ]
    ].replace([np.inf, -np.inf], np.nan).fillna(0)

    if len(out) < 30:
        out["isolation_anomaly"] = 0.0
        return out

    scaled_features = StandardScaler().fit_transform(features)
    if not np.any(scaled_features):
        out["isolation_anomaly"] = 0.0
        return out

    pca = PCA(n_components=0.95, svd_solver="full")
    transformed_features = pca.fit_transform(scaled_features)
    component_count = transformed_features.shape[1]
    component_columns = [
        f"ml_pca_component_{component + 1}"
        for component in range(component_count)
    ]
    out[component_columns] = transformed_features
    out.attrs["ml_pca_explained_variance_ratio"] = float(
        pca.explained_variance_ratio_.sum()
    )

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
    )
    model.fit(transformed_features)

    raw = -model.decision_function(transformed_features)
    out["isolation_anomaly"] = safe_minmax(
        pd.Series(raw, index=out.index)
    )

    return out


def generate_reason(row: pd.Series) -> str:
    reasons = []

    if float(row["amount_anomaly"]) >= 0.60:
        reasons.append(
            "amount is large compared with normal activity in its account"
        )

    if int(row["historical_similar_count"]) == 0:
        reasons.append(
            "no similar historical transaction was found"
        )
    elif int(row["historical_years_with_similar_activity"]) >= 2:
        reasons.append(
            "similar activity appears across prior years"
        )

    if float(row["semantic_novelty"]) >= 0.65:
        reasons.append(
            "description is unusual within the ledger"
        )

    if int(row["description_count"]) <= 1:
        reasons.append(
            "description is rare in this account"
        )

    if (
        str(row.get("vendor", "")).strip() != ""
        and int(row["vendor_count"]) <= 1
    ):
        reasons.append(
            "vendor is new or rare for this account"
        )

    if int(row["days_from_year_end"]) <= 30:
        reasons.append(
            "posted close to the reporting-date period"
        )

    if not reasons:
        reasons.append("multiple weaker signals combined")

    return "; ".join(reasons)


def assign_review_priority(scores: pd.Series) -> pd.Series:
    return pd.cut(
        scores,
        bins=[-np.inf, *REVIEW_PRIORITY_THRESHOLDS, np.inf],
        labels=["Low", "Medium", "High", "Very High"],
        right=False,
    )


def assign_review_bucket(row: pd.Series) -> str:
    score = float(row.get("review_score", 0.0))
    recurrence = float(row.get("historical_recurrence_strength", 0.0))

    if recurrence >= 0.67 and score >= 0.25:
        return "Likely recurring"
    if score >= 0.50:
        return "Manual investigation"
    if score >= 0.25:
        return "Monitor"
    return "Ignore"


def get_review_score_band(score: float) -> str:
    if score >= 0.75:
        return "Very high score band"
    if score >= 0.50:
        return "High score band"
    if score >= 0.25:
        return "Medium score band"
    return "Low score band"


def build_review_summary(row: pd.Series) -> dict[str, str | int]:
    score = float(row.get("review_score", 0.0))
    reasons = [
        part.strip()
        for part in str(row.get("why_flagged", "")).split(";")
        if part.strip()
    ]

    return {
        "priority": str(row.get("review_priority", "Low")),
        "score_band": get_review_score_band(score),
        "reason_count": len(reasons),
        "top_reason": reasons[0] if reasons else "No strong review signal",
    }


def build_signal_breakdown(row: pd.Series) -> list[dict[str, float | str]]:
    signals = [
        "amount_anomaly",
        "historical_recurrence_anomaly",
        "semantic_novelty",
        "vendor_novelty",
        "frequency_rarity",
        "year_end_context",
        "journal_anomaly",
        "isolation_anomaly",
    ]

    amount_penalty = (
        0.20
        if float(row.get("abs_amount", MIN_ABS_AMOUNT)) < MIN_ABS_AMOUNT
        else 1.0
    )
    weights = {
        "amount_anomaly": 0.85 * WEIGHTS["amount_anomaly"],
        "historical_recurrence_anomaly": 0.85 * WEIGHTS["historical_recurrence"],
        "semantic_novelty": 0.85 * WEIGHTS["semantic_novelty"],
        "vendor_novelty": 0.85 * WEIGHTS["vendor_novelty"],
        "frequency_rarity": 0.85 * WEIGHTS["frequency_rarity"],
        "year_end_context": 0.85 * WEIGHTS["year_end_context"],
        "journal_anomaly": 0.85 * WEIGHTS["journal_anomaly"],
        "isolation_anomaly": 0.15,
    }

    breakdown = []
    for signal in signals:
        value = float(row.get(signal, 0.0))
        contribution = weights[signal] * value * amount_penalty

        breakdown.append({
            "signal": signal,
            "value": round(value, 4),
            "contribution": round(contribution, 4),
        })

    breakdown.sort(key=lambda item: item["contribution"], reverse=True)
    return breakdown


def build_dominant_signal(row: pd.Series) -> dict[str, float | str]:
    breakdown = build_signal_breakdown(row)
    dominant = breakdown[0] if breakdown else {"signal": "unknown", "value": 0.0, "contribution": 0.0}

    labels = {
        "amount_anomaly": "large amount relative to account norms",
        "historical_recurrence_anomaly": "weak historical recurrence evidence",
        "semantic_novelty": "description is unusually novel in this ledger",
        "vendor_novelty": "vendor is new or rare for this account",
        "frequency_rarity": "description occurs infrequently in this account",
        "year_end_context": "transaction falls near the reporting cut-off",
        "journal_anomaly": "journal pattern differs from normal activity",
        "isolation_anomaly": "outlier pattern compared with similar transactions",
    }

    return {
        "signal": str(dominant["signal"]),
        "label": labels.get(str(dominant["signal"]), "unusual activity signal"),
        "contribution": float(dominant["contribution"]),
        "value": float(dominant["value"]),
    }


def add_final_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    has_vendor = out["vendor"].fillna("").astype(str).str.strip().ne("")
    out["vendor_novelty"] = 0.0
    if has_vendor.any():
        out.loc[has_vendor, "vendor_novelty"] = safe_minmax(
            1.0 / out.loc[has_vendor, "vendor_count"].clip(lower=1)
        )

    core = (
        WEIGHTS["amount_anomaly"] * out["amount_anomaly"]
        + WEIGHTS["historical_recurrence"] * out[
            "historical_recurrence_anomaly"
        ]
        + WEIGHTS["semantic_novelty"] * out["semantic_novelty"]
        + WEIGHTS["vendor_novelty"] * out["vendor_novelty"]
        + WEIGHTS["frequency_rarity"] * out["frequency_rarity"]
        + WEIGHTS["year_end_context"] * out["year_end_context"]
        + WEIGHTS["journal_anomaly"] * out["journal_anomaly"]
    )

    out["review_score"] = (
        0.85 * core + 0.15 * out["isolation_anomaly"]
    )

    out.loc[
        out["abs_amount"] < MIN_ABS_AMOUNT,
        "review_score"
    ] *= 0.20

    out["review_priority"] = assign_review_priority(out["review_score"])
    out["review_bucket"] = out.apply(assign_review_bucket, axis=1)

    out["why_flagged"] = out.apply(
        generate_reason, axis=1
    )
    out["review_score_band"] = out["review_score"].apply(get_review_score_band)
    out["review_summary"] = out.apply(build_review_summary, axis=1)
    out["signal_breakdown"] = out.apply(build_signal_breakdown, axis=1)
    out["dominant_signal"] = out.apply(build_dominant_signal, axis=1)
    out["dominant_signal_label"] = out["dominant_signal"].apply(
        lambda value: value.get("label", "unusual activity signal")
        if isinstance(value, dict) else "unusual activity signal"
    )

    out["model_interpretation"] = np.where(
        out["historical_recurrence_strength"] >= 0.67,
        "Evidence of recurring activity — do not treat as non-recurring "
        "without further analysis",
        np.where(
            out["review_score"] >= 0.50,
            "Potentially non-recurring — analyst review recommended",
            "Insufficient evidence for high-priority review",
        ),
    )

    return out.sort_values(
        ["review_score", "abs_amount"],
        ascending=[False, False]
    ).reset_index(drop=True)


def _run_pipeline_loaded(
    df: pd.DataFrame,
    expense_only: bool = True,
    fiscal_year_end_month: int = 12,
) -> pd.DataFrame:
    load_summary = {
        "source_row_count": df.attrs.get("source_row_count", len(df)),
        "dropped_row_count": df.attrs.get("dropped_row_count", 0),
    }
    df = classify_expense_rows(df)

    expense_filter_fallback = False
    if expense_only:
        expense_df = df[df["is_expense_candidate"]].copy()

        if len(expense_df) >= max(10, int(len(df) * 0.02)):
            df = expense_df
        else:
            expense_filter_fallback = True

    df = add_journal_features(df)
    df = add_frequency_features(df)
    df = add_amount_anomaly(df)
    df = add_year_end_context(
        df,
        fiscal_year_end_month=fiscal_year_end_month,
    )
    df = add_historical_recurrence(df)
    df = add_semantic_novelty(df)
    df = add_isolation_forest(df)
    result = add_final_score(df)
    result.attrs.update(load_summary)
    result.attrs["expense_filter_fallback"] = expense_filter_fallback
    return result


def run_pipeline(
    path: str | Path,
    expense_only: bool = True,
    fiscal_year_end_month: int = 12,
) -> pd.DataFrame:
    return _run_pipeline_loaded(
        load_gl(path),
        expense_only=expense_only,
        fiscal_year_end_month=fiscal_year_end_month,
    )


def run_pipeline_frames(
    frames: Iterable[tuple[str, pd.DataFrame]],
    expense_only: bool = True,
    fiscal_year_end_month: int = 12,
) -> pd.DataFrame:
    loaded_frames = []
    source_row_count = 0
    dropped_row_count = 0

    for source_index, (source_name, frame) in enumerate(frames):
        loaded = load_gl(frame)
        source_row_count += int(loaded.attrs.get("source_row_count", len(loaded)))
        dropped_row_count += int(loaded.attrs.get("dropped_row_count", 0))
        loaded["source_file"] = str(source_name)
        loaded["journal_id"] = (
            f"{source_index}:{source_name}::" + loaded["journal_id"].astype(str)
        )
        loaded_frames.append(loaded)

    if not loaded_frames:
        raise ValueError("Upload at least one ledger file.")

    combined = pd.concat(loaded_frames, ignore_index=True, sort=False)
    combined.attrs["source_row_count"] = source_row_count
    combined.attrs["dropped_row_count"] = dropped_row_count
    return _run_pipeline_loaded(
        combined,
        expense_only=expense_only,
        fiscal_year_end_month=fiscal_year_end_month,
    )


def save_outputs(
    result: pd.DataFrame,
    output_path: str | Path = "gl_analysis_results.csv",
    review_path: str | Path = "analyst_review.csv",
    top_n: int = DEFAULT_TOP_N,
) -> tuple[Path, Path]:

    output_path = Path(output_path)
    review_path = Path(review_path)

    result.to_csv(output_path, index=False)

    review_columns = [
        "source_file",
        "date",
        "journal_id",
        "account",
        "sub_account",
        "vendor",
        "description",
        "amount",
        "review_score",
        "review_priority",
        "why_flagged",
        "model_interpretation",
        "description_count",
        "historical_similar_count",
        "historical_years_with_similar_activity",
        "historical_recurrence_strength",
        "amount_anomaly",
        "semantic_novelty",
        "isolation_anomaly",
        "days_from_year_end",
        "journal_line_count",
        "journal_total_abs",
        "journal_id_is_synthetic",
    ]

    review_columns = [
        c for c in review_columns if c in result.columns
    ]

    review = result[
        result["review_score"] >= 0.25
    ][review_columns].head(top_n)

    review.to_csv(review_path, index=False)

    return output_path, review_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Screen a General Ledger for potentially "
            "non-recurring expenses."
        )
    )
    parser.add_argument(
        "input_file",
        help="Path to CSV/XLSX/XLS GL file"
    )
    parser.add_argument(
        "--all-accounts",
        action="store_true",
        help="Analyse all accounts instead of expense-like accounts only."
    )
    args = parser.parse_args()

    result = run_pipeline(
        args.input_file,
        expense_only=not args.all_accounts
    )

    output_path, review_path = save_outputs(result)

    print("\nAnalysis complete")
    print(f"Transactions analysed: {len(result):,}")
    print(
        "High / Very High priority: "
        f"{result['review_priority'].isin(['High', 'Very High']).sum():,}"
    )
    print(f"Full results: {output_path}")
    print(f"Analyst review list: {review_path}")

    print("\nTop candidates:\n")
    cols = [
        "date", "account", "description",
        "amount", "review_score",
        "review_priority", "why_flagged"
    ]
    print(result[cols].head(20).to_string(index=False))
