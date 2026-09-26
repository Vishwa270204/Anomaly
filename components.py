"""Rendering helpers. Every function here takes already-computed data and
draws it -- no data crunching happens in this module."""
import html
import re
import textwrap

import streamlit as st

_STATUS_STYLE = {
    "PASS": ("check-pass", "✓"),
    "OK": ("check-pass", "✓"),
    "WARN": ("check-warn", "!"),
    "REVIEW": ("check-warn", "!"),
    "FAIL": ("check-fail", "✕"),
}


def render_check_rows(checks, name_key="name"):
    """Render a list of {name/rule, status, detail} dicts as check rows.
    Shared by data-quality checks and AI evidence-consistency checks so
    both look and behave identically."""
    for check in checks:
        status = str(check.get("status", "WARN")).upper()
        icon_class, icon = _STATUS_STYLE.get(status, ("check-warn", "!"))
        name = html.escape(str(check.get(name_key, check.get("name", check.get("rule", "")))))
        detail = html.escape(str(check.get("detail", "")))
        st.markdown(
            f'<div class="check-row"><div class="check-icon {icon_class}">{icon}</div>'
            f'<div><div class="check-name">{name}</div>'
            f'<div class="check-detail">{detail}</div></div></div>',
            unsafe_allow_html=True,
        )


def _worst_status(checks):
    statuses = {str(c.get("status", "")).upper() for c in checks}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses or "REVIEW" in statuses:
        return "WARN"
    return "PASS"


def render_key_observations(obs):
    if obs["anomaly_count"] == 0:
        st.caption("No anomalies in the selected period.")
        return

    lines = [f"**{obs['anomaly_count']:,}** anomalous observations across **{obs['event_count']}** persistent event(s)."]
    if obs["typical_duration"] is not None:
        lines.append(f"Typical persistent-event duration: **~{obs['typical_duration']:.0f} min**.")
    if obs["max_severity"] is not None:
        lines.append(f"Maximum observed anomaly score (not a probability): **{obs['max_severity']:.2f}**.")
    if obs["top_features"]:
        lines.append("Variables contributing most to unusual reconstruction error: **" + ", ".join(obs["top_features"]) + "**.")
    if obs["off_baseline_metrics"]:
        lines.append(
            "Outside the typical healthy range during anomalies: **" + ", ".join(obs["off_baseline_metrics"])
            + "** (supporting evidence only, not proof of cause)."
        )
    for line in lines:
        st.markdown(f"- {line}")


def render_ai_explanation(data):
    """Render the structured LLM explanation as an alert banner followed by
    Why / When / Recommended-action sections."""
    if not data:
        return

    def esc(value):
        return html.escape(str(value)) if value is not None else ""

    when = data.get("when_occurred") if isinstance(data.get("when_occurred"), dict) else {}
    why_html = "".join(f"<li>{esc(b)}</li>" for b in data.get("why_it_happened", []) if b)
    action_html = "".join(f"<li>{esc(b)}</li>" for b in data.get("recommended_actions", []) if b)
    operating_pattern = esc(when.get("operating_pattern", ""))
    operating_row = (
        f'<div class="ai-row"><div class="ai-label">Operating pattern</div><div class="ai-body">{operating_pattern}</div></div>'
        if operating_pattern else ""
    )

    block = f'''
        <div class="ai-card">
            <div class="ai-alert-box">
                <div class="ai-alert-icon">⚠️</div>
                <div>
                    <div class="ai-alert-title">{esc(data.get("headline", ""))}</div>
                    <div class="ai-alert-desc">{esc(data.get("summary", ""))}</div>
                </div>
            </div>
            <div class="ai-section">
                <div class="ai-section-header"><span class="ai-section-icon">🔍</span> Why it happened?</div>
                <ul class="ai-bullets">{why_html}</ul>
            </div>
            <div class="ai-section">
                <div class="ai-section-header"><span class="ai-section-icon">🕐</span> When it occurred?</div>
                <div class="ai-row"><div class="ai-label">Time pattern</div><div class="ai-body">{esc(when.get("time_pattern", "—"))}</div></div>
                <div class="ai-row"><div class="ai-label">Duration pattern</div><div class="ai-body">{esc(when.get("duration_pattern", "—"))}</div></div>
                {operating_row}
            </div>
            <div class="ai-section">
                <div class="ai-section-header"><span class="ai-section-icon">🔧</span> Recommended action</div>
                <ul class="ai-bullets">{action_html}</ul>
            </div>
            <div class="ai-footer-note">
                <span>ℹ️</span>
                <span>Based on patterns observed across the selected anomaly population and the
                trained anomaly detection model. This does not confirm a fault, but helps you
                understand where to look first.</span>
            </div>
        </div>
        '''
    # Markdown's raw-HTML block parser ends a block at the first blank line;
    # after that, any leftover indentation reads as a code block. Stripping
    # all leading whitespace keeps the whole card as one HTML block.
    block = re.sub(r"(?m)^[ \t]+", "", textwrap.dedent(block)).strip()
    st.markdown(block, unsafe_allow_html=True)


def render_evidence_consistency(checks):
    """AI evidence-consistency checks, collapsed inside an expander whose
    title already reflects the worst status found."""
    if not checks:
        return
    label = "Passed" if _worst_status(checks) == "PASS" else "Review"
    with st.expander(f"Evidence consistency: {label}", expanded=False):
        st.caption(
            "These checks confirm the explanation's claims are consistent with the "
            "supplied data. They do not validate a physical diagnosis."
        )
        render_check_rows(checks, name_key="name")


def render_event_validation(results, summary):
    """Data-validation results for one selected anomaly event."""
    if results is None:
        st.caption("Data validation module is not available in this deployment.")
        return

    if summary:
        pill_class = {"PASS": "validation-pass", "NEEDS_REVIEW": "validation-warn", "FAIL": "validation-fail"}.get(summary["verdict"], "validation-warn")
        st.markdown(
            f'<div class="validation-header">'
            f'<div class="validation-title">Evidence check</div>'
            f'<div class="validation-pill {pill_class}">{summary["verdict"].replace("_", " ")}</div>'
            f'</div>'
            f'<div class="validation-subtitle">{summary["pass_count"]} passed · {summary["warn_count"]} to review · {summary["fail_count"]} failed</div>',
            unsafe_allow_html=True,
        )
    render_check_rows(results, name_key="rule")
    st.markdown(
        '<div class="validation-note">Validates the data behind this event (timing, direction, '
        "numbers, and available thresholds) — not the physical root cause.</div>",
        unsafe_allow_html=True,
    )
