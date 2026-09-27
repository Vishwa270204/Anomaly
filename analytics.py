"""Data loading and analytics.

Everything here is pure data-crunching: reading the parquet files the
notebook produces, detecting persistent anomaly events, comparing anomalies
to the healthy baseline, and assembling the evidence structures that are
shared between what the dashboard displays and what gets sent to the AI /
validation layer (so the numbers on screen and the numbers the AI sees can
never disagree).
"""
from datetime import datetime

import pandas as pd
import streamlit as st

from utils import clean_value, get_contribution_cols

# Metrics used for baseline comparisons and evidence. Kept as one list so the
# comparison table, the AI evidence, and the healthy-baseline summary all
# agree on the same set of variables.
COMPARISON_METRICS = [
    ("dc_power_kw", "DC Power", " kW"),
    ("ac_power_kw", "AC Power", " kW"),
    ("inverter_temperature_c", "Temperature", " °C"),
    ("efficiency_pct", "Efficiency", "%"),
    ("power_factor", "Power Factor", ""),
    ("dc_current_a", "DC Current", " A"),
]

EVIDENCE_NUMERIC_COLUMNS = [
    "dc_power_kw", "dc_current_a", "ac_power_kw", "ac_current_a",
    "power_factor", "frequency_hz", "efficiency_pct",
    "inverter_temperature_c", "ambient_temperature_c",
    "poa_w_m2", "ghi_w_m2", "packet_loss_pct", "communication_latency_ms",
]

EVENT_MIN_DURATION_MIN = 60
SAMPLE_INTERVAL_MIN = 5
EVENT_MIN_POINTS = int(EVENT_MIN_DURATION_MIN / SAMPLE_INTERVAL_MIN)
EVENT_MAX_GAP_MIN = SAMPLE_INTERVAL_MIN * 1.5


# ============================================================
# LOADING (cached — parquet files are read once per session)
# ============================================================

@st.cache_data
def load_dashboard_data(path="dashboard_data.parquet"):
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["anomaly_flag"] = df["anomaly_flag"].fillna(False).astype(bool) if "anomaly_flag" in df.columns else False
    return df


@st.cache_data
def load_dashboard_baseline(path="dashboard_baseline.parquet"):
    baseline = pd.read_parquet(path)
    numeric_cols = [c for c in baseline.columns if c.endswith(("_median", "_q10", "_q90"))]
    for col in numeric_cols:
        baseline[col] = pd.to_numeric(baseline[col], errors="coerce")
    if "healthy_sample_count" in baseline.columns:
        baseline["healthy_sample_count"] = pd.to_numeric(baseline["healthy_sample_count"], errors="coerce")
    return baseline


def safe_load(loader, path, label):
    """Load a required parquet file, stopping the app with a readable error
    instead of crashing on a missing/broken file."""
    try:
        return loader(path)
    except FileNotFoundError:
        st.error(
            f"**{label} not found** (`{path}`). Run `inverter_anomaly.ipynb` "
            "to generate it, then place it next to `app.py`."
        )
        st.stop()
    except Exception as e:
        st.error(f"Failed to load **{label}** (`{path}`): {e}")
        st.stop()


# ============================================================
# HEALTHY BASELINE
# ============================================================

def aggregate_baseline_reference(baseline_df, metrics):
    """Median-of-medians reference values for each metric across all
    healthy baseline groups — a population-level reference, not matched to
    any single timestamp's operating conditions."""
    reference = {}
    if baseline_df is None or baseline_df.empty:
        return reference
    for metric in metrics:
        median_col, q10_col, q90_col = f"{metric}_median", f"{metric}_q10", f"{metric}_q90"
        if median_col not in baseline_df.columns:
            continue
        med = pd.to_numeric(baseline_df[median_col], errors="coerce").dropna()
        if med.empty:
            continue
        q10 = pd.to_numeric(baseline_df.get(q10_col), errors="coerce").dropna()
        q90 = pd.to_numeric(baseline_df.get(q90_col), errors="coerce").dropna()
        reference[metric] = {
            "median": clean_value(med.median()),
            "typical_low_q10": clean_value(q10.median()) if not q10.empty else None,
            "typical_high_q90": clean_value(q90.median()) if not q90.empty else None,
        }
    return reference


def compare_population_to_baseline(anomaly_df, baseline_reference):
    """Compare mean anomaly-population values against the aggregated healthy
    baseline reference. Supporting evidence only — does not establish cause."""
    rows = []
    if anomaly_df is None or anomaly_df.empty:
        return rows
    for metric, label, unit in COMPARISON_METRICS:
        if metric not in anomaly_df.columns or metric not in baseline_reference:
            continue
        values = pd.to_numeric(anomaly_df[metric], errors="coerce").dropna()
        if values.empty:
            continue
        observed = values.mean()
        ref = baseline_reference[metric]
        low, high = ref.get("typical_low_q10"), ref.get("typical_high_q90")
        if low is not None and high is not None:
            if observed < low:
                status = "Below typical range"
            elif observed > high:
                status = "Above typical range"
            else:
                status = "Within typical range"
            range_str = f"{low:.1f}–{high:.1f}{unit}"
        else:
            status = "No healthy reference available"
            range_str = "—"
        rows.append({
            "label": label,
            "observed": f"{observed:.1f}{unit}",
            "typical_range": range_str,
            "status": status,
        })
    return rows


# ============================================================
# KEY OBSERVATIONS
# ============================================================

def build_key_observations(anomaly_df, events_df, comparison_rows):
    """Operator-facing summary answering 'what should I investigate first?'.
    Built only from values computed elsewhere — no new claims."""
    obs = {
        "anomaly_count": int(len(anomaly_df)),
        "event_count": int(len(events_df)),
        "typical_duration": None,
        "max_severity": None,
        "top_features": [],
        "off_baseline_metrics": [],
    }
    if not events_df.empty and "duration_min" in events_df.columns:
        durations = pd.to_numeric(events_df["duration_min"], errors="coerce").dropna()
        if not durations.empty:
            obs["typical_duration"] = float(durations.median())

    if "anomaly_score_ratio" in anomaly_df.columns:
        scores = pd.to_numeric(anomaly_df["anomaly_score_ratio"], errors="coerce").dropna()
        if not scores.empty:
            obs["max_severity"] = float(scores.max())

    contribution_cols = get_contribution_cols(anomaly_df)
    if contribution_cols:
        means = (
            anomaly_df[contribution_cols].apply(pd.to_numeric, errors="coerce")
            .mean().dropna().sort_values(ascending=False)
        )
        obs["top_features"] = [c.replace("_contribution_pct", "") for c in means.head(3).index]

    obs["off_baseline_metrics"] = [
        r["label"] for r in comparison_rows if r["status"] in ("Above typical range", "Below typical range")
    ]
    return obs


# ============================================================
# ANOMALY EVENTS
# ============================================================

def build_anomaly_events(anomaly_df):
    """Group anomalous readings into persistent events.

    Rule: an event needs at least EVENT_MIN_DURATION_MIN of continuous
    anomalous readings. At a 5-minute sampling interval that's 12
    consecutive anomaly points; a gap greater than 7.5 minutes breaks a run.
    """
    if anomaly_df.empty:
        return pd.DataFrame()

    work = anomaly_df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()

    gap_min = work["timestamp"].diff().dt.total_seconds() / 60.0
    run_id = (gap_min.isna() | (gap_min > EVENT_MAX_GAP_MIN)).cumsum()
    run_sizes = run_id.value_counts()
    persistent_runs = run_sizes[run_sizes >= EVENT_MIN_POINTS].index
    work = work[run_id.isin(persistent_runs)].copy()
    work["run_id"] = run_id[run_id.isin(persistent_runs)]
    if work.empty:
        return pd.DataFrame()

    events = []
    for run, rows in work.groupby("run_id"):
        rows = rows.sort_values("timestamp").reset_index(drop=True)
        start_time, end_time = rows["timestamp"].min(), rows["timestamp"].max()
        event = {
            "start_time": start_time,
            "end_time": end_time,
            "duration_min": (end_time - start_time).total_seconds() / 60.0 + SAMPLE_INTERVAL_MIN,
            "anomaly_count": len(rows),
        }
        if "anomaly_score_ratio" in rows.columns:
            scores = pd.to_numeric(rows["anomaly_score_ratio"], errors="coerce")
            event["max_severity"], event["mean_severity"] = scores.max(), scores.mean()
        if "inverter_status" in rows.columns:
            mode = rows["inverter_status"].mode()
            event["dominant_status"] = mode.iloc[0] if len(mode) else None
        if "anomaly_type" in rows.columns:
            mode = rows["anomaly_type"].mode()
            event["anomaly_type"] = mode.iloc[0] if len(mode) else None
        events.append(event)

    events_df = pd.DataFrame(events).sort_values("start_time").reset_index(drop=True)
    events_df.insert(0, "event_id", [f"Event {i + 1}" for i in range(len(events_df))])
    return events_df


# ============================================================
# EVIDENCE STRUCTURES
# (shared by the AI generator, the AI evidence-consistency validator, and
#  the standalone data_validation module)
# ============================================================

def build_population_evidence(anomaly_df, events_df, baseline_df, start_date, end_date, selected_inverter):
    """One evidence structure for the whole anomaly population in scope."""
    evidence = {
        "analysis_scope": {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "inverter": "All" if selected_inverter == "All" else str(selected_inverter),
        },
        "anomaly_summary": {
            "anomaly_observation_count": int(len(anomaly_df)),
            "persistent_event_count": int(len(events_df)),
        },
    }

    event_patterns = {}
    if not events_df.empty:
        for col in ["duration_min", "anomaly_count", "max_severity", "mean_severity"]:
            if col in events_df.columns:
                values = pd.to_numeric(events_df[col], errors="coerce").dropna()
                if not values.empty:
                    event_patterns[col] = {
                        "min": clean_value(values.min()), "mean": clean_value(values.mean()),
                        "median": clean_value(values.median()), "max": clean_value(values.max()),
                    }
        event_patterns["earliest_event_start"] = clean_value(events_df["start_time"].min())
        event_patterns["latest_event_end"] = clean_value(events_df["end_time"].max())
    evidence["event_patterns"] = event_patterns

    operating_patterns = {}
    for col in ["inverter_status", "is_daylight", "quality_code", "communication_status", "fault_code", "alarm_code"]:
        if col not in anomaly_df.columns:
            continue
        values = anomaly_df[col].dropna().astype(str)
        if not values.empty:
            operating_patterns[col] = {str(k): int(v) for k, v in values.value_counts().head(10).items()}
    evidence["operating_patterns"] = operating_patterns

    contribution_cols = get_contribution_cols(anomaly_df)
    if contribution_cols:
        means = (
            anomaly_df[contribution_cols].apply(pd.to_numeric, errors="coerce")
            .mean().dropna().sort_values(ascending=False)
        )
        evidence["feature_contributions"] = [
            {"feature": col.replace("_contribution_pct", ""), "mean_contribution_pct": clean_value(value)}
            for col, value in means.head(10).items()
        ]
    else:
        evidence["feature_contributions"] = []

    stats = {}
    for col in EVIDENCE_NUMERIC_COLUMNS:
        if col not in anomaly_df.columns:
            continue
        values = pd.to_numeric(anomaly_df[col], errors="coerce").dropna()
        if not values.empty:
            stats[col] = {
                "min": clean_value(values.min()), "mean": clean_value(values.mean()),
                "median": clean_value(values.median()), "max": clean_value(values.max()),
            }
    evidence["anomaly_observations"] = {"observation_count": int(len(anomaly_df)), "statistics": stats}

    baseline_reference = aggregate_baseline_reference(baseline_df, EVIDENCE_NUMERIC_COLUMNS)
    evidence["healthy_baseline"] = {"reference": baseline_reference}
    evidence["baseline_comparisons"] = compare_population_to_baseline(anomaly_df, baseline_reference)

    time_patterns = {}
    if "timestamp" in anomaly_df.columns:
        ts = pd.to_datetime(anomaly_df["timestamp"], errors="coerce").dropna()
        if not ts.empty:
            time_patterns["earliest_anomaly"] = clean_value(ts.min())
            time_patterns["latest_anomaly"] = clean_value(ts.max())
            time_patterns["hour_counts"] = {str(int(k)): int(v) for k, v in ts.dt.hour.value_counts().sort_index().items()}
    evidence["time_patterns"] = time_patterns

    return evidence


def build_event_evidence(event, df, baseline_reference):
    """Evidence for ONE selected anomaly event, in the shape the
    data_validation module expects: event window/duration, the start/end
    statistics claimed for that window, and any healthy-baseline reference
    available for comparison.
    """
    start, end = event["start_time"], event["end_time"]
    if "anomaly_flag" in df.columns:
        rows = df[df["anomaly_flag"] & (df["timestamp"] >= start) & (df["timestamp"] <= end)]
    else:
        rows = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

    statistics = {}
    for col in ("dc_power_kw", "ac_power_kw", "inverter_temperature_c"):
        if col not in rows.columns:
            continue
        series = pd.to_numeric(rows[col], errors="coerce").dropna()
        if not series.empty:
            statistics[col] = {"start": clean_value(series.iloc[0]), "end": clean_value(series.iloc[-1])}

    healthy_reference = {
        metric: baseline_reference[metric]
        for metric in ("inverter_temperature_c",)
        if metric in baseline_reference
    }

    # data_validation.py recomputes duration as the raw (end - start) gap,
    # so the claimed duration here must match that -- not the dashboard's
    # own "+1 sample interval" coverage-adjusted duration used elsewhere.
    raw_duration_min = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / 60.0

    return {
        "event": {
            "start_time": str(start),
            "end_time": str(end),
            "duration_min": clean_value(raw_duration_min),
        },
        "event_observations": {"statistics": statistics},
        "healthy_baseline": {"reference": healthy_reference},
    }
