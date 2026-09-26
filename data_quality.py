"""Optional integration with data_validation.py.

data_validation.py checks the evidence for ONE selected anomaly event
against the source data (event window, power direction, numeric evidence,
temperature/threshold claims). It's an optional module: if it isn't
present next to app.py, event verification degrades gracefully instead of
crashing the dashboard.
"""
try:
    from data_validation import validate_event_data, summarize
    HAS_DATA_VALIDATION = True
except Exception:
    HAS_DATA_VALIDATION = False
    validate_event_data = summarize = None


def verify_event(evidence, df):
    """Run the standalone validator against one event's evidence.

    Returns (results, summary). Both are None if the module isn't
    available; a validation failure is reported as a single WARN check
    rather than raising, so a bug in the optional module never crashes
    the dashboard.
    """
    if not HAS_DATA_VALIDATION:
        return None, None
    try:
        results = validate_event_data(evidence, df)
        return results, summarize(results)
    except Exception as exc:
        return [{"rule": "Data validation", "status": "WARN", "detail": f"Validation could not run: {exc}"}], None
