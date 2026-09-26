"""Small, dependency-free helpers shared by the analytics, AI, and UI layers.

Keeping these in one place means every module cleans/formats values the
same way instead of re-implementing slightly different versions.
"""
from datetime import datetime

import pandas as pd


def clean_value(value):
    """Convert a pandas/numpy scalar into a plain, JSON-friendly Python value.

    Returns None for NaN/NaT instead of raising, so callers can pass the
    result straight into json.dumps or an f-string.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def fmt_num(value, decimals=2, suffix="", dash="—"):
    """Format a possibly-missing numeric value; never raises on NaN/None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return dash
    return f"{value:,.{decimals}f}{suffix}"


def get_contribution_cols(frame):
    """Feature-contribution columns exported from the training notebook,
    named '<feature>_contribution_pct'."""
    return [c for c in frame.columns if c.endswith("_contribution_pct")]
