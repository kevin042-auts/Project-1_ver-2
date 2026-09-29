
import calendar
import logging
from html import escape

import pandas as pd
import streamlit as st

from gl_normalization_v2 import run_pipeline_frames
from upload_utils import (
    fingerprint_uploaded_files,
    normalize_uploaded_files,
    read_uploaded_ledger,
)


logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Ledger Review | Expense Screening",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def analyze_ledger(
    frames: tuple[tuple[str, pd.DataFrame], ...],
    expense_only: bool,
    fiscal_year_end_month: int,
) -> pd.DataFrame:
    return run_pipeline_frames(
        frames,
        expense_only=expense_only,
        fiscal_year_end_month=fiscal_year_end_month,
    )


def record_analyst_decision(
    decisions_key: str,
    transaction_index: str,
    decision: str,
) -> None:
    st.session_state[decisions_key][transaction_index] = decision


st.markdown(
    """
    <style>
    :root {
        --ink: #1a2430;
        --muted: #475467;
        --accent: #0d5c4e;
        --accent-soft: #eaf4f1;
        --canvas: #f5f7f7;
        --line: #d7e0dd;
        --panel: #ffffff;
        --warning: #8a5a00;
        --warning-soft: #fff6df;
    }
    .stApp {
        color: var(--ink);
        background-color: var(--canvas);
        font-family: 'Segoe UI', sans-serif;
        letter-spacing: 0;
    }
    .stApp [data-testid="stMarkdownContainer"] p { color: var(--muted); }
    .stApp [data-testid="stCaptionContainer"] p,
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stCheckbox"] label,
    .stApp [data-testid="stRadio"] label { color: var(--muted); }
    .stApp [data-testid="stMarkdownContainer"] p.eyebrow { color: var(--accent); }
    .stApp [data-baseweb="select"] * { color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: #eef1ef;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
    }
    .main .block-container {
        max-width: 1380px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    .sidebar-title {
        margin: 0.4rem 0 2rem; color: var(--ink);
        font: 700 14px 'Segoe UI', sans-serif; letter-spacing: 0.3px;
    }
    .app-masthead {
        margin: 0 0 1.3rem; padding: 8px 0 18px 18px;
        color: var(--ink); background: transparent; border-left: 3px solid var(--accent);
    }
    .masthead-top { display: flex; align-items: center; gap: 11px; margin-bottom: 20px; }
    .masthead-mark {
        display: grid; place-items: center; width: 36px; height: 36px;
        color: #fff; background: var(--accent); border-radius: 3px;
        font: 700 12px 'Segoe UI', sans-serif; letter-spacing: 0.7px;
    }
    .masthead-brand { color: var(--ink); font: 700 13px 'Segoe UI', sans-serif; }
    .masthead-context { color: var(--muted); font: 600 11px 'Segoe UI', sans-serif; letter-spacing: 1px; }
    .masthead-title { margin: 0; color: var(--ink); font: 650 30px 'Aptos Display', 'Segoe UI', sans-serif; }
    .stApp .masthead-description { margin: 5px 0 0; color: var(--muted); font-size: 14px; }
    .section-lead { display: flex; align-items: end; justify-content: space-between; margin: 0 0 14px; }
    .section-lead h2 { margin: 0; color: var(--ink); font: 650 22px 'Aptos Display', 'Segoe UI', sans-serif; }
    .section-lead p { margin: 4px 0 0; color: var(--muted); font-size: 13px; }
    .step-label { color: var(--accent); font: 700 11px 'Segoe UI', sans-serif; letter-spacing: 1px; }
    .eyebrow {
        margin: 0 0 0.35rem; color: var(--accent);
        font: 700 11px 'Segoe UI', sans-serif; letter-spacing: 1.2px;
    }
    .section-heading {
        display: flex; align-items: end; justify-content: space-between;
        margin: 1.4rem 0 0.8rem; padding-bottom: 0.6rem;
        border-bottom: 1px solid var(--line);
    }
    .section-heading h2 { margin: 0; color: var(--ink); font: 650 21px 'Aptos Display', 'Segoe UI', sans-serif; }
    .section-count {
        padding: 4px 9px; color: var(--accent); background: var(--accent-soft);
        border-radius: 4px; font: 600 12px 'Segoe UI', sans-serif;
    }
    .upload-guide { padding: 0.9rem 0.75rem; border-left: 2px solid var(--accent); }
    .upload-guide h3 { margin: 0.35rem 0 0.9rem; color: var(--ink); font: 650 18px 'Aptos Display', 'Segoe UI', sans-serif; }
    .upload-guide p { margin: 0 0 0.8rem; color: var(--muted); font-size: 14px; line-height: 1.6; }
    .upload-guide strong { color: var(--ink); }
    .upload-title { margin: 0 0 12px; color: var(--ink); font: 650 19px 'Aptos Display', 'Segoe UI', sans-serif; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #fff; border-color: var(--line); border-radius: 6px;
    }
    [data-testid="stMetric"] {
        padding: 14px 16px; background: var(--panel);
        border: 1px solid var(--line); border-radius: 5px;
    }
    [data-testid="stMetricLabel"] p { color: var(--muted); font-size: 12px; }
    [data-testid="stMetricValue"] { color: var(--ink); }
    [data-testid="stFileUploaderDropzone"] {
        min-height: 120px; background: #f9fbfa; border: 1px dashed #66736d;
    }
    [data-testid="stFileUploaderDropzone"] * { color: var(--ink); }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 5px; }
    .review-shell { margin-top: 0.35rem; }
    .review-stage {
        display: grid; gap: 1rem; grid-template-columns: 1.3fr 0.7fr;
        align-items: start;
    }
    .summary-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbfa 100%);
        border: 1px solid var(--line); border-radius: 10px;
        padding: 1rem 1.05rem; height: 100%; min-height: 176px;
        box-shadow: 0 1px 2px rgba(10, 27, 22, 0.04);
    }
    .summary-card .label {
        color: var(--muted); font: 700 11px 'Segoe UI', sans-serif; letter-spacing: 0.9px; text-transform: uppercase;
        margin-bottom: 0.55rem;
    }
    .summary-card .value {
        color: var(--ink); font: 700 20px 'Segoe UI', sans-serif; line-height: 1.2;
        word-break: break-word;
    }
    .summary-card .meta {
        color: var(--muted); margin-top: 0.45rem; font-size: 13px; line-height: 1.5;
    }
    .tag-row { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.8rem; }
    .tag {
        display: inline-block; padding: 5px 8px; border-radius: 999px;
        background: var(--accent-soft); color: var(--accent); border: 1px solid #cfe2dc;
        font: 600 11px 'Segoe UI', sans-serif; letter-spacing: 0.2px;
    }
    .compact-table .stDataFrame { font-size: 12px; }
    [data-testid="stDataFrame"] thead th { font-weight: 700; }
    @media (max-width: 900px) {
        .review-stage { grid-template-columns: 1fr; }
    }
    div.stButton > button, div.stDownloadButton > button {
        min-height: 42px; border-radius: 4px; font-weight: 600;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        color: #10261f !important; background: linear-gradient(180deg, #ffd76a 0%, #f7b42d 100%);
        border-color: #d79a1a; box-shadow: 0 0 0 1px rgba(183, 124, 15, 0.08);
        font-weight: 700;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        color: #10261f !important; background: linear-gradient(180deg, #f6ce5b 0%, #e79b16 100%);
        border-color: #c6810d;
    }
    div[data-testid="stButton"] button[kind="primary"] span,
    div[data-testid="stButton"] button[kind="primary"] div {
        color: #10261f !important;
    }
    div[data-testid="stDownloadButton"] button[kind="secondary"] {
        color: var(--accent); background: #fff; border: 1px solid var(--accent);
        font-weight: 600;
    }
    div[data-testid="stDownloadButton"] button[kind="secondary"]:hover {
        color: #075c40; background: var(--accent-soft); border-color: #075c40;
    }
    @media (max-width: 700px) {
        .main .block-container { padding: 1.25rem 1rem 2rem; }
        .app-masthead { padding: 8px 0 18px 14px; }
        .masthead-title { font-size: 26px; }
        .section-lead { align-items: start; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">LEDGER REVIEW</div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Analysis settings**")
    expense_only = st.checkbox(
        "Analyse expense-related accounts only",
        value=True,
        help=(
            "Recommended for this project. The app uses account type when "
            "available and otherwise applies an account-name heuristic."
        ),
    )
    fiscal_year_end_month = st.selectbox(
        "Fiscal year-end month",
        options=range(1, 13),
        index=11,
        format_func=lambda month: calendar.month_name[month],
    )
    st.caption("Screening support only. An analyst must review flagged items.")

st.markdown(
    '<div class="app-masthead">'
    '<div class="masthead-top"><div class="masthead-mark">GL</div>'
    '<div><div class="masthead-brand">LEDGER REVIEW</div>'
    '<div class="masthead-context">ML-ASSISTED EXPENSE SCREENING</div></div></div>'
    '<h1 class="masthead-title">Expense review</h1>'
    '<p class="masthead-description">Use machine-learning signals to prioritize potentially non-recurring expenses for analyst review.</p>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-lead"><div><h2>Start a review</h2>'
    '<p>Upload your source ledger to begin transaction screening.</p></div></div>',
    unsafe_allow_html=True,
)

upload_col, guidance_col = st.columns([1.35, 1])
with upload_col:
    with st.container(border=True):
        st.markdown(
            '<p class="eyebrow">SOURCE FILE</p>'
            '<h2 class="upload-title">Upload a ledger</h2>',
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "General ledger files",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
        )
        st.caption("Accepted formats: CSV, XLSX, XLS · multiple files supported")
        if uploaded_files:
            st.caption(f"Selected files: {len(uploaded_files)}")

with guidance_col:
    st.markdown(
        '<div class="upload-guide"><p class="eyebrow">BEFORE YOU BEGIN</p>'
        '<h3>Prepare your source</h3>'
        '<p><strong>Required fields</strong><br>date, account, description, and amount '
        '(or debit and credit).</p>'
        '<p><strong>Recommended</strong><br>Include multiple years to improve historical recurrence context.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

normalized_uploads = normalize_uploaded_files(uploaded_files)
if not normalized_uploads:
    st.stop()

upload_id = fingerprint_uploaded_files(normalized_uploads)
analysis_id = f"{upload_id}_{int(expense_only)}_{fiscal_year_end_month}"
decisions_key = f"analyst_decisions_{analysis_id}"
analyst_decisions = st.session_state.setdefault(decisions_key, {})
audit_log_key = f"audit_log_{analysis_id}"
st.session_state.setdefault(audit_log_key, {})

try:
    with st.spinner("Analysing the General Ledger..."):
        frames = tuple(
            (uploaded_file.name, read_uploaded_ledger(uploaded_file))
            for uploaded_file in normalized_uploads
        )
        result = analyze_ledger(
            frames,
            expense_only=expense_only,
            fiscal_year_end_month=fiscal_year_end_month,
        )
        result = result.sort_values(
            ["review_score", "abs_amount"],
            ascending=[False, False],
        ).reset_index(drop=True)
        result["analyst_decision"] = [
            analyst_decisions.get(str(index), "Unreviewed")
            for index in result.index
        ]

    uploaded_names = [file.name for file in normalized_uploads]
    if len(uploaded_names) == 1:
        st.caption(f"Analysis ready · {uploaded_names[0]}")
    else:
        st.caption(f"Analysis ready · {len(uploaded_names)} files merged into one review queue")
    dropped_row_count = int(result.attrs.get("dropped_row_count", 0))
    source_row_count = int(result.attrs.get("source_row_count", len(result)))
    if dropped_row_count:
        st.warning(
            f"Excluded {dropped_row_count:,} of {source_row_count:,} source rows "
            "because date, amount, account, or description was invalid or missing."
        )

    total_transactions = len(result)
    very_high = int(
        (result["review_priority"] == "Very High").sum()
    )
    recurring_warning = int(
        (result["historical_recurrence_strength"] >= 0.67).sum()
    )

    review = result[
        result["review_score"] >= 0.25
    ].copy()
    displayed_review_count = min(len(review), 100)
    priority_cell_styles = {
        "Very High": "background-color: #fbe9e7; color: #8f261e; font-weight: 700;",
        "High": "background-color: #fff0e3; color: #984b0a; font-weight: 700;",
        "Medium": "background-color: #fff6d8; color: #735500; font-weight: 600;",
        "Low": "background-color: #e8f2ed; color: #245744;",
    }

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transactions analysed", f"{total_transactions:,}")
    m2.metric("Items in review queue", f"{len(review):,}")
    m3.metric("Very-high priority", f"{very_high:,}")
    m4.metric("Recurring-pattern warnings", f"{recurring_warning:,}")
    pca_columns = [
        column
        for column in result.columns
        if column.startswith("ml_pca_component_")
    ]
    if pca_columns:
        explained_variance = result.attrs.get("ml_pca_explained_variance_ratio")
        variance_text = (
            f"; {explained_variance:.1%} of scaled-feature variance retained"
            if explained_variance is not None
            else ""
        )
        st.caption(
            f"ML anomaly transformation: StandardScaler -> PCA "
            f"({len(pca_columns)} components{variance_text}) "
            "-> Isolation Forest."
        )
    else:
        st.caption(
            "PCA and Isolation Forest are skipped when fewer than 30 "
            "transactions are available or model features have no variation."
        )

    if "expense_filter_fallback" in result.attrs and result.attrs["expense_filter_fallback"]:
        st.info(
            "Expense-only filtering produced too few candidates for this ledger, so the review was expanded to maintain a usable review queue."
        )

    table_cols = [
        "date",
        "account",
        "description",
        "amount",
        "review_score",
        "review_priority",
        "dominant_signal_label",
        "analyst_decision",
    ]
    if len(normalized_uploads) > 1:
        table_cols.insert(0, "source_file")

    st.markdown(
        '<div class="section-heading"><div><p class="eyebrow">REVIEW QUEUE</p>'
        '<h2>Analyst review queue</h2></div>'
        f'<span class="section-count">'
        f'Showing {displayed_review_count:,} of {len(review):,}</span></div>',
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="compact-table">', unsafe_allow_html=True)
        review_table = review[table_cols].head(100).style.map(
            lambda priority: priority_cell_styles.get(str(priority), ""),
            subset=["review_priority"],
        )
        st.dataframe(
            review_table,
            use_container_width=True,
            hide_index=True,
            height=360,
            column_config={
                "source_file": st.column_config.TextColumn("Source file", width="small"),
                "date": st.column_config.DateColumn("Posting date", format="MMM D, YYYY"),
                "account": st.column_config.TextColumn("GL account", width="small"),
                "description": st.column_config.TextColumn("Transaction description", width="large"),
                "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                "review_score": st.column_config.NumberColumn(
                    "Review score",
                    format="%.2f",
                    help="Higher values indicate stronger screening signals; this is not a probability.",
                ),
                "review_priority": st.column_config.TextColumn("Review priority"),
                "dominant_signal_label": st.column_config.TextColumn("Main reason flagged"),
                "analyst_decision": st.column_config.TextColumn("Analyst decision"),
            },
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading"><div><p class="eyebrow">TRANSACTION DETAIL</p>'
        '<h2>Transaction evidence</h2></div></div>',
        unsafe_allow_html=True,
    )

    if review.empty:
        st.info("No transactions crossed the review threshold.")
    else:
        with st.container(border=True):
            def format_transaction(index):
                source = (
                    f"{review.loc[index, 'source_file']} | "
                    if len(normalized_uploads) > 1 else ""
                )
                return (
                    f"{source}{review.loc[index, 'date'].date()} | "
                    f"{review.loc[index, 'account']} | "
                    f"${review.loc[index, 'amount']:,.2f} | "
                    f"{review.loc[index, 'description']}"
                )

            selected_transaction = st.selectbox(
                "Selected transaction",
                options=review.index.tolist(),
                index=0,
                format_func=format_transaction,
                key=f"selected_transaction_{analysis_id}",
            )
            row = result.loc[selected_transaction]

            a, b, c, d = st.columns(4)

            a.metric("Review score", f"{row['review_score']:.2f}")
            b.metric("Priority", str(row["review_priority"]))
            c.metric("Bucket", str(row["review_bucket"]))
            d.metric("Prior active years", int(row["historical_years_with_similar_activity"]))

            st.markdown('<div class="review-shell">', unsafe_allow_html=True)
            summary_col, detail_col = st.columns([0.92, 1.08])

            with summary_col:
                st.markdown(
                    """
                    <div class="summary-card">
                        <div class="label">Transaction summary</div>
                        <div class="value">{account}</div>
                        <div class="meta">{description}<br>{date} · ${amount:,.2f}</div>
                        <div class="tag-row">
                            <span class="tag">{priority}</span>
                            <span class="tag">{bucket}</span>
                            <span class="tag">{score_band}</span>
                        </div>
                    </div>
                    """.format(
                        account=escape(str(row["account"])),
                        description=escape(str(row["description"])),
                        date=escape(
                            row["date"].strftime("%Y-%m-%d")
                            if hasattr(row["date"], "strftime")
                            else str(row["date"])
                        ),
                        amount=float(row["amount"]),
                        priority=escape(str(row["review_priority"])),
                        bucket=escape(str(row["review_bucket"])),
                        score_band=escape(
                            str(row.get("review_score_band", "Low score band"))
                        ),
                    ),
                    unsafe_allow_html=True,
                )

            with detail_col:
                st.markdown("**Score band and rationale**")
                summary = row.get("review_summary", {})
                if isinstance(summary, dict):
                    st.write(f"Top reason: {summary.get('top_reason', 'No strong review signal')}")
                dominant_signal = row.get("dominant_signal", {})
                if isinstance(dominant_signal, dict):
                    st.write(f"Dominant signal: {dominant_signal.get('signal', 'unknown')} — {dominant_signal.get('label', 'unusual activity signal')}")
                signal_breakdown = row.get("signal_breakdown", [])
                if signal_breakdown:
                    signal_df = pd.DataFrame(signal_breakdown)
                    signal_df = signal_df[["signal", "value", "contribution"]]
                    st.markdown("**Score contribution breakdown**")
                    st.dataframe(signal_df, use_container_width=True, hide_index=True, height=220)
                st.write(row["why_flagged"])
                interpretation = row["model_interpretation"]
                if "recurring activity" in interpretation.lower():
                    st.info(interpretation)
                elif row["review_score"] >= 0.50:
                    st.warning(interpretation)
                else:
                    st.caption(interpretation)

            st.markdown('</div>', unsafe_allow_html=True)

            evidence = pd.DataFrame(
                {
                    "Signal": [
                        "Amount anomaly",
                        "Historical recurrence anomaly",
                        "Description novelty",
                        "Frequency rarity",
                        "Vendor novelty",
                        "Year-end context",
                        "Journal anomaly",
                        "Isolation Forest anomaly",
                    ],
                    "Value": [
                        round(float(row["amount_anomaly"]), 3),
                        round(float(row["historical_recurrence_anomaly"]), 3),
                        round(float(row["semantic_novelty"]), 3),
                        round(float(row["frequency_rarity"]), 3),
                        round(float(row["vendor_novelty"]), 3),
                        int(row["year_end_context"]),
                        round(float(row["journal_anomaly"]), 3),
                        round(float(row["isolation_anomaly"]), 3),
                    ],
                }
            )

            st.markdown("**Evidence signals**")
            st.dataframe(
                evidence,
                use_container_width=True,
                hide_index=True,
                height=220,
            )

            if pca_columns:
                with st.expander("ML transformation details"):
                    st.caption(
                        "PCA components are learned from standardized ledger "
                        "features and used by Isolation Forest. They are model "
                        "inputs, not probabilities or accounting amounts."
                    )
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Component": pca_columns,
                                "Value": [
                                    round(float(row[column]), 3)
                                    for column in pca_columns
                                ],
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

            st.markdown("**Analyst decision**")

            c1, c2, c3 = st.columns([1, 1, 1.5])

            with c1:
                st.button(
                    "Keep",
                    key=f"keep_{analysis_id}_{selected_transaction}",
                    help="Record this transaction as reviewed with no change.",
                    on_click=record_analyst_decision,
                    args=(decisions_key, str(selected_transaction), "KEEP"),
                    use_container_width=True,
                )
            with c2:
                st.button(
                    "Adjust",
                    key=f"adjust_{analysis_id}_{selected_transaction}",
                    help="Mark this item as needing a correction; this does not edit the ledger.",
                    on_click=record_analyst_decision,
                    args=(decisions_key, str(selected_transaction), "ADJUST"),
                    use_container_width=True,
                )
            with c3:
                st.button(
                    "Investigate",
                    type="primary",
                    icon=":material/search:",
                    help="Flag this transaction for follow-up investigation.",
                    key=f"investigate_{analysis_id}_{selected_transaction}",
                    on_click=record_analyst_decision,
                    args=(decisions_key, str(selected_transaction), "INVESTIGATE"),
                    use_container_width=True,
                )

            decision = analyst_decisions.get(str(selected_transaction), "Unreviewed")
            if decision != "Unreviewed":
                st.caption(f"Recorded decision: {decision}")

                audit_entry_key = str(selected_transaction)
                if audit_entry_key not in st.session_state[audit_log_key]:
                    audit_note = {
                        "transaction_index": int(selected_transaction),
                        "account": str(row["account"]),
                        "amount": float(row["amount"]),
                        "priority": str(row["review_priority"]),
                        "bucket": str(row["review_bucket"]),
                        "decision": decision,
                    }
                    st.session_state[audit_log_key][audit_entry_key] = audit_note

            st.caption(
                "Decisions are retained for this upload during the current session "
                "and included in the downloaded CSV."
            )

    result["analyst_decision"] = [
        analyst_decisions.get(str(index), "Unreviewed")
        for index in result.index
    ]
    csv_bytes = result.to_csv(index=False).encode("utf-8")

    st.divider()
    st.caption("Export the full analysis and recorded decisions as a CSV.")
    st.download_button(
        "Download full analysis",
        data=csv_bytes,
        file_name="gl_analysis_results.csv",
        mime="text/csv",
        type="secondary",
        icon=":material/download:",
    )

except ValueError as exc:
    logger.info("Invalid General Ledger upload: %s", exc)
    st.error(f"Could not analyse the General Ledger: {exc}")
except Exception:
    logger.exception("Failed to analyse uploaded General Ledger")
    st.error(
        "The General Ledger could not be analysed. Check the file format "
        "and required columns, then try again."
    )
