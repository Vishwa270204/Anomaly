"""
Inverter Anomaly Detection -- Streamlit Dashboard
====================================================
Production frontend only. All ML/training happens in inverter_anomaly.ipynb.

Reads:
    dashboard_data.parquet      -> evaluation-period observations + model output
    dashboard_baseline.parquet  -> healthy operating baseline (quantiles)

Does NOT retrain or re-run the notebook. Does NOT treat anomaly_score_ratio
as a probability. Feature contributions are reported as "contributed most
to reconstruction error," never as a proven cause.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import (
    COMPARISON_METRICS,
    aggregate_baseline_reference,
    build_anomaly_events,
    build_population_evidence,
    compare_population_to_baseline,
    load_dashboard_baseline,
    load_dashboard_data,
    safe_load,
)
from ai_engine import generate_overall_ai_explanation, validate_overall_explanation
from components import render_ai_explanation
from styles import DASHBOARD_CSS
from utils import fmt_num

# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(page_title="Inverter Anomaly Detection", page_icon="⚡", layout="wide")
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

st.session_state.setdefault("ai_explanations", {})
# Drop any cached explanation that doesn't match the current response
# schema, so a schema change regenerates instead of breaking the renderer.
st.session_state["ai_explanations"] = {
    key: value
    for key, value in st.session_state["ai_explanations"].items()
    if isinstance(value, dict) and isinstance(value.get("explanation"), (dict, type(None)))
    and (value.get("explanation") is None or "when_occurred" in value["explanation"])
}

# ============================================================
# LOAD DATA
# ============================================================

df = safe_load(load_dashboard_data, "dashboard_data.parquet", "Dashboard data")
baseline_df = safe_load(load_dashboard_baseline, "dashboard_baseline.parquet", "Dashboard healthy baseline")

if df.empty:
    st.error("`dashboard_data.parquet` loaded but contains no rows.")
    st.stop()

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">
        <div class="app-header-icon">⚡</div>
        <div>
            <div class="app-header-title">Inverter Anomaly Detection</div>
            <div class="app-header-subtitle">Monitor inverter performance, investigate anomaly events, and review AI-generated explanations.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# FILTERS
# ============================================================

min_date, max_date = df["timestamp"].min().date(), df["timestamp"].max().date()
show_inverter_filter = "inverter_id" in df.columns and df["inverter_id"].nunique() > 1

with st.container(border=True):
    cols = st.columns([1, 1, 2] if show_inverter_filter else [1, 3])

    with cols[0]:
        show_anomalies_only = st.checkbox("Show anomalies only", value=False)

    if show_inverter_filter:
        with cols[1]:
            inverter_options = ["All"] + sorted(df["inverter_id"].dropna().unique().tolist())
            selected_inverter = st.selectbox("Inverter", inverter_options)
        caption_col = cols[2]
    else:
        selected_inverter = "All"
        caption_col = cols[1]

    with caption_col:
        st.caption("**Severity**: higher = further outside normal range.")

start_date, end_date = min_date, max_date

# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()

if selected_inverter != "All" and "inverter_id" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["inverter_id"] == selected_inverter].copy()
if show_anomalies_only:
    filtered_df = filtered_df[filtered_df["anomaly_flag"]].copy()

if filtered_df.empty:
    st.warning("No observations match the current filters. Adjust the filters above.")

total_observations = len(filtered_df)
total_anomalies = int(filtered_df["anomaly_flag"].sum()) if total_observations else 0
anomaly_rate = (total_anomalies / total_observations * 100) if total_observations else None
anomaly_df = filtered_df[filtered_df["anomaly_flag"]].copy() if total_observations else filtered_df.copy()

events_df = build_anomaly_events(anomaly_df)

comparison_metric_names = [m for m, _, _ in COMPARISON_METRICS]
baseline_reference = aggregate_baseline_reference(baseline_df, comparison_metric_names)
comparison_rows = compare_population_to_baseline(anomaly_df, baseline_reference)

# ============================================================
# TABS  (AI generation first, everything else second)
# ============================================================

tab_ai, tab_details = st.tabs(["AI Analysis", "Overview & Events"])

# ------------------------------------------------------------
# TAB: AI ANALYSIS
# ------------------------------------------------------------
with tab_ai:
    st.markdown("## Overall Anomaly Explanation")
    st.caption(
        "AI analysis of the recurring patterns across all detected anomalies "
        "in the selected period. No individual anomaly is selected."
    )

    if len(anomaly_df) == 0:
        st.info("No anomalies were detected in the selected period, so there is no anomaly pattern to explain.")
    else:
        header_col, button_col = st.columns([5.5, 1.5])
        header_col.markdown('<div class="ai-title">Why are anomalies occurring?</div>', unsafe_allow_html=True)
        regenerate = button_col.button(
            "Regenerate", type="primary", use_container_width=True,
            help="Generate a fresh overall explanation from all detected anomalies.",
        )

        cache_key = f"overall_{selected_inverter}_{start_date}_{end_date}"
        cached = st.session_state["ai_explanations"].get(cache_key)

        if regenerate or cached is None:
            evidence = build_population_evidence(anomaly_df, events_df, baseline_df, start_date, end_date, selected_inverter)
            with st.spinner("Analyzing all detected anomalies..."):
                try:
                    explanation = generate_overall_ai_explanation(evidence)
                    checks = validate_overall_explanation(explanation, evidence)
                    cached = {"explanation": explanation, "checks": checks, "error": None}
                except Exception as e:
                    cached = {"explanation": None, "checks": None, "error": str(e)}
            st.session_state["ai_explanations"][cache_key] = cached

        if cached.get("error"):
            st.error(f"AI explanation failed: {cached['error']}")
        elif cached.get("explanation"):
            checks = cached.get("checks") or []
            has_fail = any(c.get("status") == "FAIL" for c in checks)
            if has_fail:
                st.error(
                    "The AI explanation was withheld because one or more claims contradict "
                    "the supplied evidence. Try regenerating."
                )
            else:
                render_ai_explanation(cached["explanation"])
        else:
            st.warning("AI did not return an overall explanation.")

# ------------------------------------------------------------
# TAB: OVERVIEW & EVENTS  (everything else)
# ------------------------------------------------------------
with tab_details:
    st.markdown("## Overview")
    st.caption("A high-level view of inverter observations and detected anomalies for the selected period.")

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Observations", f"{total_observations:,}")
    kpi_cols[1].metric("Anomalous Observations", f"{total_anomalies:,}")
    kpi_cols[2].metric("Anomaly Rate", fmt_num(anomaly_rate, 2, "%"))
    kpi_cols[3].metric("Persistent Events", f"{len(events_df):,}")

    if total_observations > 0 and total_anomalies == 0:
        st.caption("✅ No anomalies found in this period — everything looks normal.")

    st.markdown("### Anomaly Score Over Time")
    st.caption(
        "Higher points mean more unusual behavior relative to the trained model. "
        "Red dots are flagged anomalies; shaded bands mark persistent events."
    )

    if total_observations > 0 and "reconstruction_error" in filtered_df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_df["timestamp"], y=filtered_df["reconstruction_error"],
            mode="lines", name="Anomaly Score", line=dict(color="#4C78A8", width=1.5),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Anomaly score: %{y:.3f}<extra></extra>",
        ))
        anomaly_points = filtered_df[filtered_df["anomaly_flag"]]
        if len(anomaly_points) > 0:
            fig.add_trace(go.Scatter(
                x=anomaly_points["timestamp"], y=anomaly_points["reconstruction_error"],
                mode="markers", name="Flagged anomaly", marker=dict(size=8, color="#E45756"),
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Anomaly score: %{y:.3f}<extra>Flagged</extra>",
            ))
        for _, ev in events_df.iterrows():
            fig.add_vrect(x0=ev["start_time"], x1=ev["end_time"], fillcolor="#E45756", opacity=0.08, line_width=0)

        fig.update_layout(
            xaxis_title="Time", yaxis_title="Anomaly Score", yaxis_type="log",
            hovermode="x unified", height=380, template="plotly_white",
            margin=dict(t=20, l=55, r=25, b=45),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, width="stretch")
        if not events_df.empty:
            st.caption("Shaded bands mark persistent anomaly events (≥60 min of continuous anomalous readings).")
    else:
        st.info("No anomaly score data available for the selected period.")

    st.markdown("### Anomalies vs Healthy Baseline")
    st.caption(
        "Average values observed during anomalies compared with the typical healthy "
        "range. Supporting evidence only — it does not prove a fault or its cause."
    )
    if comparison_rows:
        comparison_table = pd.DataFrame(comparison_rows).rename(columns={
            "label": "Variable", "observed": "Observed during anomalies",
            "typical_range": "Typical healthy range", "status": "Status",
        })
        st.dataframe(comparison_table, width="stretch", hide_index=True)
    else:
        st.caption("No overlapping variables between the anomaly data and the healthy baseline.")

    st.markdown("## Detected Anomaly Events")
    st.caption("Persistent anomaly events detected in the selected period (≥60 minutes of continuous anomalous readings).")

    if len(events_df) == 0:
        if total_observations > 0:
            st.success("✅ No anomaly events were detected in this period.")
        else:
            st.info("No readings in this date range. Try a different range above.")
    else:
        display_columns = [
            "event_id", "start_time", "end_time", "duration_min", "anomaly_count",
            "max_severity", "mean_severity", "dominant_status", "anomaly_type",
        ]
        display_columns = [c for c in display_columns if c in events_df.columns]
        friendly_names = {
            "event_id": "Event", "start_time": "Start Time", "end_time": "End Time",
            "duration_min": "Duration (min)", "anomaly_count": "Anomaly Points",
            "max_severity": "Max Severity", "mean_severity": "Mean Severity",
            "dominant_status": "Dominant Status", "anomaly_type": "Dominant Anomaly Type",
        }
        table = events_df[display_columns].rename(columns=friendly_names)
        for col in ["Max Severity", "Mean Severity", "Duration (min)"]:
            if col in table.columns:
                table[col] = pd.to_numeric(table[col], errors="coerce").round(2)

        st.dataframe(table, width="stretch", hide_index=True)
        st.caption("Severity values reflect the anomaly score, not a probability of failure.")
