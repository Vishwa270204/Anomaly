"""AI explanation layer.

generate_overall_ai_explanation() asks an LLM (via Groq) to describe the
recurring pattern across all detected anomalies in scope, as a small
structured JSON object.

validate_overall_explanation() is an evidence-consistency gate: it checks
that every checkable claim in the explanation is actually supported by the
evidence that was sent to the model. It is NOT a physical diagnosis
validator — it never judges whether the explanation is the right one,
only whether its claims are grounded in the supplied numbers.
"""
import json
import os
import re

import pandas as pd
import streamlit as st
from groq import Groq

GROQ_MODEL = "openai/gpt-oss-20b"

REQUIRED_FIELDS = {
    "headline": str,
    "summary": str,
    "why_it_happened": list,
    "when_occurred": dict,
    "recommended_actions": list,
}
REQUIRED_WHEN_FIELDS = ("time_pattern", "duration_pattern", "operating_pattern")

BANNED_PHRASES = [
    "root cause", "probability of", "confidence score", "proven cause",
    "confirmed fault", "caused by", "caused the", "caused this",
    "resulted from", "responsible for", "triggered by", "due to", "because of",
]

SYSTEM_PROMPT = (
    "You are a careful technical writer for a solar-plant monitoring dashboard. "
    "Return ONLY a single valid JSON object -- no markdown fences, no commentary "
    "before or after it."
)

EXPLANATION_JSON_SCHEMA = """{
  "headline": "One short sentence describing the overall anomaly pattern in plain, general terms.",
  "summary": "2-3 concise sentences describing the strongest evidence-backed pattern. Use actual before/after numbers from value_changes and/or quantitative baseline comparison values when useful.",
  "why_it_happened": ["evidence-backed interpretation 1", "evidence-backed interpretation 2", "evidence-backed interpretation 3"],
  "when_occurred": {
    "time_pattern": "A general description of when this tends to happen, only if supported by evidence. No specific dates or clock times.",
    "duration_pattern": "A general sense of how long or how often this tends to happen, only if supported by evidence. Avoid exact counts or precise minute figures.",
    "operating_pattern": "Recurring operating-condition patterns (status, communication, daylight), described generally, only if supported by evidence."
  },
  "recommended_actions": ["practical operator check 1", "practical operator check 2", "practical operator check 3"]
}"""

EXPLANATION_RULES = """Rules:
- Write like a plant operator summarizing a recurring pattern to a colleague.
- Treat the supplied EVIDENCE as the only source of factual claims. Do not invent
  measurements, operating states, causes, dates, counts, or comparisons.
- The "value_changes" evidence contains typical start -> end readings across
  persistent anomaly events. Use these concrete numbers when they help explain
  the observed pattern. The evidence also provides change_percent; use it only
  when it is directly supported by the supplied values.
- The "baseline_comparisons" evidence is the preferred quantitative comparison
  between anomalous observations and the healthy baseline. It contains observed
  mean/median, healthy median, q10/q90 range, difference_percent, and status.
  When mentioning a baseline difference, use the supplied direction and values.
- The "healthy_baseline" object is a population-level reference from the
  baseline parquet. It is NOT matched to a timestamp and does not prove a fault.
- "why_it_happened" must contain evidence-backed interpretations, not asserted
  physical causes. Say "consistent with", "associated with", "may indicate", or
  "warrants investigation" when appropriate. If the evidence cannot establish
  a physical cause, say so explicitly.
- Feature contributions indicate variables associated with unusual reconstruction
  error. They do not prove physical root cause and must never be presented as a
  confirmed physical cause.
- Never use causal or diagnostic wording such as "root cause", "caused by",
  "because of", "due to", "triggered by", "confirmed fault", or "proven cause".
- Never call anomaly_score_ratio or reconstruction_error a probability or confidence.
- Do NOT cite exact event counts, timestamps, dates, or precise durations. If duration
  or frequency is useful, use conversational wording such as "about an hour" or
  "repeatedly", but only when supported by evidence.
- Do not expose ML implementation terms such as autoencoder, reconstruction error,
  threshold, probability, or confidence in operator-facing text.
- Prefer patterns supported by multiple observations/events over isolated readings.
- Recommended actions should be inspection/verification steps supported by the
  observed evidence. Do not claim that a component has failed.
- Keep every field concise and grounded only in the supplied evidence."""



@st.cache_resource
def get_groq_client():
    """One Groq client per Streamlit session/process."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["groq"]["api_key"]
        except Exception:
            api_key = None
    return Groq(api_key=api_key) if api_key else None


def _extract_json(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start:end + 1]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"The AI returned malformed JSON ({exc}).") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("The AI returned valid JSON, but it was not a JSON object.")
    return parsed


def generate_overall_ai_explanation(evidence):
    """Generate one evidence-based explanation for the whole anomaly
    population in scope (not a single selected event)."""
    client = get_groq_client()
    if client is None:
        raise RuntimeError("GROQ_API_KEY is not configured. Set it in the environment or Streamlit secrets.")

    user_prompt = (
        "Analyze ALL detected anomalies in the supplied period as one population. "
        "Describe recurring patterns across the group, not a single event.\n\n"
        f"Return ONLY valid JSON with this exact structure:\n{EXPLANATION_JSON_SCHEMA}\n\n"
        f"{EXPLANATION_RULES}\n\n"
        f"EVIDENCE:\n{json.dumps(evidence, default=str, ensure_ascii=False)}"
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=1600,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    return _extract_json(raw)


# ============================================================
# EVIDENCE-CONSISTENCY VALIDATION
# ============================================================

def validate_overall_explanation(explanation, evidence):
    """Check every checkable claim in `explanation` against `evidence`.
    Hard contradictions are FAIL; claims that can't be checked are WARN."""
    checks = []

    def add(name, status, detail):
        checks.append({"name": name, "status": status, "detail": detail})

    structure_errors = _check_structure(explanation)
    add(
        "Response structure",
        "FAIL" if structure_errors else "PASS",
        "; ".join(structure_errors) if structure_errors
        else "All required explanation fields are present with the expected types.",
    )

    when = explanation.get("when_occurred") if isinstance(explanation.get("when_occurred"), dict) else {}
    full_text, lower_text = _flatten_text(explanation, when)

    _check_terminology(lower_text, add)
    _check_event_count(when, evidence, add)
    _check_durations(when, evidence, add)
    _check_day_night_pattern(when, evidence, add)
    _check_operating_conditions(when, evidence, add)
    _check_feature_contributions(lower_text, evidence, add)
    _check_metric_values(lower_text, evidence, add)
    _check_value_change_claims(lower_text, evidence, add)
    _check_baseline_direction(lower_text, evidence, add)
    _check_baseline_numeric_claims(lower_text, evidence, add)

    return checks


def _check_structure(explanation):
    errors = []
    for key, expected_type in REQUIRED_FIELDS.items():
        if key not in explanation:
            errors.append(f"missing '{key}'")
        elif not isinstance(explanation[key], expected_type):
            errors.append(f"'{key}' has type {type(explanation[key]).__name__}, expected {expected_type.__name__}")
    when = explanation.get("when_occurred")
    if isinstance(when, dict):
        for key in REQUIRED_WHEN_FIELDS:
            if key not in when or not isinstance(when.get(key), str):
                errors.append(f"'when_occurred.{key}' is missing or not a string")
    return errors


def _flatten_text(explanation, when):
    parts = [str(explanation.get("headline", "")), str(explanation.get("summary", ""))]
    parts += [str(x) for x in explanation.get("why_it_happened", []) if x]
    parts += [str(x) for x in explanation.get("recommended_actions", []) if x]
    parts += [str(when.get(k, "")) for k in REQUIRED_WHEN_FIELDS]
    full_text = " ".join(parts)
    return full_text, full_text.lower()


def _check_terminology(lower_text, add):
    found = [p for p in BANNED_PHRASES if p in lower_text]
    add(
        "Terminology safety",
        "FAIL" if found else "PASS",
        "Unsupported phrasing found: " + ", ".join(found) if found
        else "No unsupported causal, probability, or diagnosis language detected.",
    )


def _check_event_count(when, evidence, add):
    expected = evidence.get("anomaly_summary", {}).get("persistent_event_count")
    text = str(when.get("duration_pattern", "")).lower()
    mentioned = [int(x) for x in re.findall(r"\b(\d+)\s+(?:persistent\s+)?events?\b", text)]

    if expected is None or not mentioned:
        # The prompt deliberately asks the AI to avoid stating exact counts,
        # so an absent count is expected, not a gap to flag.
        add("Persistent event count", "PASS", "No exact persistent-event count was stated (expected -- the explanation is written in general terms).")
    else:
        bad = [x for x in mentioned if x != int(expected)]
        if bad:
            add("Persistent event count", "FAIL", f"AI mentions {bad[0]} persistent event(s), but evidence contains {int(expected)}.")
        else:
            add("Persistent event count", "PASS", f"Matches the {int(expected)} persistent event(s) in scope.")


def _check_durations(when, evidence, add):
    duration_stats = evidence.get("event_patterns", {}).get("duration_min", {})
    known_values = [float(v) for v in duration_stats.values() if isinstance(v, (int, float)) and pd.notna(v)] \
        if isinstance(duration_stats, dict) else []

    text = str(when.get("duration_pattern", ""))
    claims = []
    for raw, unit in re.findall(r"(?<![A-Za-z])(-?\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|min)\b", text, re.I):
        claims.append(float(raw) * (60.0 if unit.lower().startswith(("hour", "hr")) else 1.0))

    if claims and known_values:
        bad = [v for v in claims if not any(abs(v - k) <= max(5.0, abs(k) * 0.05) for k in known_values)]
        if bad:
            add("Duration values", "FAIL", "Duration claim(s) don't match the supplied event statistics: " + ", ".join(f"{v:g} min" for v in bad[:5]) + ".")
        else:
            add("Duration values", "PASS", "Reported durations match the supplied event duration statistics.")
    elif claims:
        add("Duration values", "WARN", "The explanation gives numeric durations, but no event-duration statistics are available.")
    else:
        add("Duration values", "PASS", "No numeric duration claim requiring verification was made.")


def _daylight_counts(evidence):
    counts = evidence.get("operating_patterns", {}).get("is_daylight", {})
    day = sum(int(v) for k, v in counts.items() if str(k).lower() in {"true", "1", "yes"}) if isinstance(counts, dict) else 0
    night = sum(int(v) for k, v in counts.items() if str(k).lower() in {"false", "0", "no"}) if isinstance(counts, dict) else 0
    return day, night


def _check_day_night_pattern(when, evidence, add):
    day, night = _daylight_counts(evidence)
    text = str(when.get("time_pattern", "")).lower()
    claims = []
    if re.search(r"(?:during )?(?:daylight|daytime|sunlight)\b", text):
        claims.append("day")
    if re.search(r"(?:during )?(?:night|nighttime)\b", text):
        claims.append("night")

    if claims and (day + night) > 0:
        is_day_majority = day >= night
        contradicts = ("day" in claims and not is_day_majority) or ("night" in claims and is_day_majority)
        if contradicts:
            add("Day/night pattern", "FAIL", f"AI describes a {', '.join(claims)} pattern, but anomalies are predominantly {'daylight' if is_day_majority else 'night'}.")
        else:
            add("Day/night pattern", "PASS", "Day/night wording is consistent with the anomaly distribution.")
    elif claims:
        add("Day/night pattern", "WARN", "A day/night claim was made, but the supplied evidence has no usable daylight distribution.")
    else:
        add("Day/night pattern", "PASS", "No explicit day/night pattern claim was made.")


OPERATING_SYNONYMS = {
    "offline": {"offline", "off", "stopped", "shutdown"},
    "online": {"online", "on", "running", "run"},
    "fault": {"fault", "faulted", "error"},
    "alarm": {"alarm", "warning", "warn"},
}
OPERATING_FIELDS = ("inverter_status", "quality_code", "communication_status", "fault_code", "alarm_code")


def _check_operating_conditions(when, evidence, add):
    operating = evidence.get("operating_patterns", {})
    text = str(when.get("operating_pattern", "")).lower()

    present_values = set()
    for field in OPERATING_FIELDS:
        counts = operating.get(field, {}) if isinstance(operating, dict) else {}
        if isinstance(counts, dict):
            present_values.update(str(v).strip().lower() for v in counts.keys() if str(v).strip())

    unsupported = [
        synonym for synonym, aliases in OPERATING_SYNONYMS.items()
        if re.search(rf"(?<![A-Za-z]){synonym}(?![A-Za-z])", text)
        and not any(alias == v or alias in v for alias in aliases for v in present_values)
    ]

    if unsupported:
        add("Operating-condition claims", "FAIL", "Unsupported operating-condition claim(s): " + ", ".join(unsupported[:5]) + ".")
    else:
        add("Operating-condition claims", "PASS", "No unsupported categorical operating-condition claim was detected.")


FEATURE_ALIASES = {
    "inverter temperature": "inverter_temperature_c", "ambient temperature": "ambient_temperature_c",
    "dc power": "dc_power_kw", "ac power": "ac_power_kw", "dc current": "dc_current_a",
    "ac current": "ac_current_a", "power factor": "power_factor", "frequency": "frequency_hz",
    "efficiency": "efficiency_pct", "poa": "poa_w_m2", "ghi": "ghi_w_m2",
    "packet loss": "packet_loss_pct", "communication latency": "communication_latency_ms",
}


def _check_feature_contributions(lower_text, evidence, add):
    items = evidence.get("feature_contributions", [])
    names = [str(i.get("feature", "")).strip().lower() for i in items if isinstance(i, dict) and i.get("feature")]
    top3 = names[:3]

    if not re.search(r"(?:top|main|key|largest|leading|primary)\s+(?:contributing|contributor|contributors|feature|features)", lower_text):
        add("Feature contribution claims", "PASS", "No unsupported main/key contributor claim was made.")
        return

    matched = [canonical for alias, canonical in FEATURE_ALIASES.items() if alias in lower_text and canonical in names]
    if not matched:
        add("Feature contribution claims", "FAIL", "Describes a main/key contributor, but names no feature supported by the supplied evidence.")
        return
    unsupported = [m for m in matched if m not in top3]
    if unsupported:
        add("Feature contribution claims", "FAIL", "Main/key contributor claim doesn't match the top supplied contributors: " + ", ".join(unsupported) + ".")
    else:
        add("Feature contribution claims", "PASS", "Main/key contributor claims match the supplied top feature contributions.")


METRIC_ALIASES = {
    "dc_power_kw": (["dc power"], "kw"), "dc_current_a": (["dc current"], "a"),
    "ac_power_kw": (["ac power"], "kw"), "ac_current_a": (["ac current"], "a"),
    "power_factor": (["power factor"], ""), "frequency_hz": (["frequency"], "hz"),
    "efficiency_pct": (["efficiency"], "%"),
    "inverter_temperature_c": (["inverter temperature", "inverter temp"], "c"),
    "ambient_temperature_c": (["ambient temperature", "ambient temp"], "c"),
    "poa_w_m2": (["poa"], "w/m2"), "ghi_w_m2": (["ghi"], "w/m2"),
    "packet_loss_pct": (["packet loss"], "%"), "communication_latency_ms": (["communication latency", "latency"], "ms"),
}
NUMERIC_CLAIM_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(kW|kw|A|a|°C|C|%|Hz|hz|ms|W/m2|w/m2)\b")
SENTENCE_BOUNDARY = ".!?;"


def _check_metric_values(lower_text, evidence, add):
    stats = evidence.get("anomaly_observations", {}).get("statistics", {})
    # dc_power_kw / ac_power_kw / inverter_temperature_c are validated as
    # before/after ranges by _check_value_change_claims instead -- checking
    # them here too would double-flag the same numbers out of context.
    skip_metrics = set(evidence.get("value_changes", {}).keys())
    failures = []

    for metric, (aliases, expected_unit) in METRIC_ALIASES.items():
        if metric in skip_metrics:
            continue
        metric_stats = stats.get(metric)
        if not isinstance(metric_stats, dict):
            continue
        expected_values = [float(v) for v in metric_stats.values() if isinstance(v, (int, float)) and pd.notna(v)]
        if not expected_values:
            continue

        for alias in aliases:
            for match in re.finditer(re.escape(alias), lower_text):
                left = max((lower_text.rfind(c, 0, match.start()) for c in SENTENCE_BOUNDARY), default=-1) + 1
                right_candidates = [lower_text.find(c, match.end()) for c in SENTENCE_BOUNDARY]
                right = min([p for p in right_candidates if p >= 0], default=len(lower_text))
                candidates = list(NUMERIC_CLAIM_RE.finditer(lower_text, left, right))
                if not candidates:
                    continue
                nearest = min(candidates, key=lambda m: min(abs(m.start() - match.end()), abs(match.start() - m.end())))
                raw, unit = nearest.groups()
                normalized_unit = unit.lower().replace("°c", "c")
                if expected_unit and normalized_unit not in {expected_unit, ""}:
                    failures.append(f"{raw} {unit} near {metric}")
                elif not any(abs(float(raw) - k) <= max(0.5, abs(k) * 0.05) for k in expected_values):
                    failures.append(f"{raw} {unit} for {metric}")

    if failures:
        add("Metric-specific numerical claims", "FAIL", "Unsupported metric/value combination(s): " + ", ".join(failures[:5]) + ".")
    else:
        add("Metric-specific numerical claims", "PASS", "Numeric metric claims match the statistics of the metric they describe.")


VALUE_CHANGE_RE = re.compile(
    r"from\s+(?:about\s+|around\s+|roughly\s+|approximately\s+|~\s*)?(-?\d+(?:\.\d+)?)\s*[a-z°/%]*\s*"
    r"(?:to|down to|up to|rising to|climbing to|falling to|dropping to)\s+"
    r"(?:about\s+|around\s+|roughly\s+|approximately\s+|~\s*)?(-?\d+(?:\.\d+)?)",
    re.I,
)


def _check_value_change_claims(lower_text, evidence, add):
    """Verify any 'X rises/falls from A to B' claim against the typical
    start/end readings computed from real event data (value_changes).
    Picks the range nearest to each metric's own mention, so a sentence
    naming two metrics with two ranges doesn't cross-match them."""
    changes = evidence.get("value_changes", {})
    if not changes:
        add("Before/after value claims", "PASS", "No before/after value evidence is available to check.")
        return

    failures, claim_found = [], False
    for metric, info in changes.items():
        aliases = METRIC_ALIASES.get(metric, ([], ""))[0]
        typical_start, typical_end = info.get("typical_start"), info.get("typical_end")
        if typical_start is None or typical_end is None:
            continue
        tolerance = max(3.0, abs(typical_start) * 0.25, abs(typical_end) * 0.25)

        for alias in aliases:
            for match in re.finditer(re.escape(alias), lower_text):
                left = max((lower_text.rfind(c, 0, match.start()) for c in SENTENCE_BOUNDARY), default=-1) + 1
                right_candidates = [lower_text.find(c, match.end()) for c in SENTENCE_BOUNDARY]
                right = min([p for p in right_candidates if p >= 0], default=len(lower_text))
                window, alias_pos = lower_text[left:right], match.start() - left
                alias_end_pos = match.end() - left
                candidates = list(VALUE_CHANGE_RE.finditer(window))
                if not candidates:
                    continue
                # Natural phrasing states a metric's own range right after its
                # name ("temperature climbs from X to Y"), so prefer the
                # nearest range that follows the alias; only fall back to a
                # preceding range if none follows (avoids grabbing a
                # different metric's range from earlier in the sentence).
                following = [c for c in candidates if c.start() >= alias_end_pos]
                nearest = min(following, key=lambda m: m.start() - alias_end_pos) if following \
                    else min(candidates, key=lambda m: alias_pos - m.end())
                claim_found = True
                claimed_start, claimed_end = float(nearest.group(1)), float(nearest.group(2))
                if abs(claimed_start - typical_start) > tolerance or abs(claimed_end - typical_end) > tolerance:
                    failures.append(
                        f"{info['label']}: claims {claimed_start:g} → {claimed_end:g}, but the typical "
                        f"reading is ~{typical_start:.1f} → ~{typical_end:.1f} {info['unit']}"
                    )

    if failures:
        add("Before/after value claims", "FAIL", "; ".join(dict.fromkeys(failures))[:400] + ("..." if len("; ".join(failures)) > 400 else "."))
    elif claim_found:
        add("Before/after value claims", "PASS", "Before/after value claims match the typical start/end readings.")
    else:
        add("Before/after value claims", "WARN", "value_changes evidence was supplied, but no before/after value claim was made.")


def _check_baseline_numeric_claims(lower_text, evidence, add):
    """Validate baseline numbers without confusing units or percentage direction.

    The previous validator treated every number in a sentence as if it were the
    same kind of quantity. That can falsely reject perfectly valid text such as
    "31.4 kW, about 40% below the healthy median": 31.4 is a power value while
    40 is a percentage. This version matches claims to the metric and unit and
    treats percentage differences by magnitude, while checking directional words
    separately in _check_baseline_direction().
    """
    rows = evidence.get("baseline_comparisons", [])
    if not isinstance(rows, list) or not rows:
        add("Baseline numerical claims", "PASS", "No baseline comparison values are available to check.")
        return

    metric_aliases = {
        "dc power": "DC Power",
        "ac power": "AC Power",
        "temperature": "Temperature",
        "efficiency": "Efficiency",
        "power factor": "Power Factor",
        "dc current": "DC Current",
    }
    # Capture the unit so 40% is not compared against 40 kW, 40 A, etc.
    number_re = re.compile(
        r"(?<![A-Za-z0-9])(-?\d+(?:\.\d+)?)\s*(kW|A|°C|C|%|Hz|ms|W/m2)?\b",
        re.I,
    )

    def expected_for_unit(row, unit):
        unit = (unit or "").lower().replace("°c", "c")
        if unit == "%":
            keys = ["difference_percent"]
        elif unit in {"kw", "a", "c", "hz", "ms", "w/m2"}:
            keys = [
                "observed_mean", "observed_median", "healthy_median",
                "healthy_q10", "healthy_q90", "difference_from_healthy_median",
            ]
        else:
            # An unqualified number is allowed only when it is close to one of
            # the metric's supplied values. This preserves compatibility with
            # concise AI wording while avoiding cross-unit percentage matches.
            keys = [
                "observed_mean", "observed_median", "healthy_median",
                "healthy_q10", "healthy_q90", "difference_from_healthy_median",
                "difference_percent",
            ]
        return [
            float(row[k]) for k in keys
            if isinstance(row.get(k), (int, float)) and pd.notna(row.get(k))
        ]

    failures = []
    checked = False

    # Split into sentences so numbers belonging to another metric elsewhere in
    # the explanation cannot be attributed to this metric.
    sentence_spans = list(re.finditer(r"[^.!?;]+(?:[.!?;]|$)", lower_text))

    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label", ""))
        label_lower = label.lower()
        aliases = [alias for alias in metric_aliases if alias in label_lower]
        if not aliases:
            continue

        alias = aliases[0]
        expected_by_unit = {}
        for unit in ("kw", "a", "c", "%", "hz", "ms", "w/m2", ""):
            expected_by_unit[unit] = expected_for_unit(row, unit)

        for span in sentence_spans:
            sentence = span.group(0)
            if alias not in sentence:
                continue

            checked = True
            claims = list(number_re.finditer(sentence))
            for claim in claims:
                raw = float(claim.group(1))
                unit = (claim.group(2) or "").lower().replace("°c", "c")
                expected = expected_by_unit.get(unit, expected_by_unit[""])
                if not expected:
                    continue

                # Percent differences are commonly written as positive magnitude
                # with directional language: "40% lower". Compare magnitude.
                candidates = expected
                if unit == "%":
                    candidates = [abs(v) for v in candidates]
                    raw_cmp = abs(raw)
                else:
                    raw_cmp = raw

                if not any(abs(raw_cmp - exp) <= max(0.5, abs(exp) * 0.05) for exp in candidates):
                    failures.append(
                        f"{label}: {raw:g}{claim.group(2) or ''} is not supported by the supplied baseline comparison values"
                    )

    if failures:
        detail = "; ".join(dict.fromkeys(failures))
        add("Baseline numerical claims", "FAIL", detail[:500] + ("..." if len(detail) > 500 else "."))
    elif checked:
        add("Baseline numerical claims", "PASS", "Numeric baseline claims match the supplied comparison values and units.")
    else:
        add("Baseline numerical claims", "PASS", "No numeric baseline claim requiring verification was made.")


def _check_baseline_direction(lower_text, evidence, add):
    """Check baseline direction in the same sentence as the metric.

    The earlier implementation searched the entire explanation. If one metric
    was above baseline and another was below baseline, a direction word for the
    first metric could accidentally satisfy the check for the second. This
    sentence-local check avoids that false positive/false negative.
    """
    rows = evidence.get("baseline_comparisons", [])
    failures, claim_found = [], False

    direction_words = {
        "above": r"above|higher|elevated|exceed|greater than|increased|higher than",
        "below": r"below|lower|reduced|decreased|less than|lower than",
    }

    sentence_spans = list(re.finditer(r"[^.!?;]+(?:[.!?;]|$)", lower_text))

    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label", ""))
        label_lower = label.lower()
        status = str(row.get("status", "")).lower()
        aliases = [alias for alias in (
            "dc power", "ac power", "temperature", "efficiency",
            "power factor", "dc current"
        ) if alias in label_lower]
        if not aliases or status not in {"above typical range", "below typical range"}:
            continue

        alias = aliases[0]
        matched_sentence = False
        for span in sentence_spans:
            sentence = span.group(0)
            if alias not in sentence:
                continue
            matched_sentence = True
            claim_found = True
            if status == "above typical range":
                if not re.search(direction_words["above"], sentence):
                    failures.append(
                        f"{label}: evidence is above typical range, but the sentence containing {label} does not state an above/higher direction"
                    )
            elif status == "below typical range":
                if not re.search(direction_words["below"], sentence):
                    failures.append(
                        f"{label}: evidence is below typical range, but the sentence containing {label} does not state a below/lower direction"
                    )

        # Mentioning the metric without a directional claim is fine; the AI is
        # not required to discuss every baseline comparison.
        if not matched_sentence:
            continue

    if failures:
        add("Healthy-baseline claims", "FAIL", "; ".join(failures[:3]))
    elif claim_found:
        add("Healthy-baseline claims", "PASS", "Baseline-direction claims are consistent with the supplied comparison.")
    else:
        add("Healthy-baseline claims", "PASS", "No unsupported baseline-direction claim was detected.")

