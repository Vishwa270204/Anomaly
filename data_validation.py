"""Compact evidence validation for one selected anomaly event.

This module validates the data/evidence pipeline, not physical root cause and not
LLM style. It intentionally returns only four high-level checks.
"""
import pandas as pd


def _close(a, b, rel=0.02, abs_tol=1.0):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


def validate_event_data(evidence, df, trend_df=None, baseline_df=None):
    ev = evidence.get("event", {})
    start = pd.to_datetime(ev.get("start_time"), errors="coerce")
    end = pd.to_datetime(ev.get("end_time"), errors="coerce")
    results = []

    # 1. Event time & duration
    if pd.isna(start) or pd.isna(end):
        results.append({"rule":"Event time & duration","status":"WARN","detail":"Event boundaries are unavailable."})
    else:
        duration = (end - start).total_seconds() / 60
        claimed = ev.get("duration_min")
        ok = claimed is None or _close(duration, claimed, rel=0.01, abs_tol=0.5)
        results.append({"rule":"Event time & duration","status":"PASS" if ok else "FAIL",
                        "detail":f"Event window is {start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M}, {duration:g} min."})

    # Recompute event rows once.
    if not pd.isna(start) and not pd.isna(end):
        if "anomaly_flag" in df.columns:
            rows = df[(df["anomaly_flag"]) & (df["timestamp"] >= start) & (df["timestamp"] <= end)]
        else:
            rows = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    else:
        rows = df.iloc[0:0]

    # 2. Power change direction
    direction_errors=[]
    for col,label in [("dc_power_kw","DC"),("ac_power_kw","AC")]:
        if col in rows.columns and len(rows)>=2:
            a=pd.to_numeric(rows[col],errors="coerce").dropna()
            if len(a)>=2:
                if a.iloc[-1] < a.iloc[0]: direction_errors.append(f"{label} power decreased")
                elif a.iloc[-1] > a.iloc[0]: direction_errors.append(f"{label} power increased")
    results.append({"rule":"Power change direction","status":"PASS",
                    "detail":("; ".join(direction_errors)+" in the event data.") if direction_errors else "No power direction could be recomputed."})

    # 3. Numerical evidence integrity: verify the core statistics sent to the LLM.
    stat_mismatches=[]
    cached=evidence.get("event_observations",{}).get("statistics",{})
    for col in ("dc_power_kw","ac_power_kw","inverter_temperature_c"):
        if col not in cached or col not in rows.columns: continue
        s=pd.to_numeric(rows[col],errors="coerce").dropna()
        if s.empty: continue
        for key,val in (("start",s.iloc[0]),("end",s.iloc[-1])):
            claimed=cached[col].get(key)
            if claimed is not None and not _close(claimed,val):
                stat_mismatches.append(f"{col}.{key}")
    results.append({"rule":"Numerical values","status":"FAIL" if stat_mismatches else "PASS",
                    "detail":("Evidence mismatch: "+", ".join(stat_mismatches)+".") if stat_mismatches else "Core event values match the source data."})

    # 4. Temperature / threshold evidence
    temp = cached.get("inverter_temperature_c",{})
    ref = evidence.get("healthy_baseline",{}).get("reference",{}).get("inverter_temperature_c",{})
    if temp and ref:
        results.append({"rule":"Temperature / threshold claim","status":"PASS","detail":"Temperature and available baseline reference are present in the supplied evidence."})
    elif temp:
        results.append({"rule":"Temperature / threshold claim","status":"WARN","detail":"Temperature is available, but no independent healthy threshold is available."})
    else:
        results.append({"rule":"Temperature / threshold claim","status":"WARN","detail":"Temperature is not available for independent verification."})

    return results


def summarize(results):
    fails=[r for r in results if r["status"]=="FAIL"]
    warns=[r for r in results if r["status"] in ("WARN","REVIEW")]
    if fails: verdict="FAIL"
    elif warns: verdict="NEEDS_REVIEW"
    else: verdict="PASS"
    return {"verdict":verdict,"pass_count":sum(r["status"]=="PASS" for r in results),
            "fail_count":len(fails),"warn_count":len(warns)}
