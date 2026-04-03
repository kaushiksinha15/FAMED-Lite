from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import json
import datetime
import os

# --- Graduated Response Tiers ---
# Each tier defines: (max_drift_share, status_label, action, description)
DRIFT_TIERS = [
    (0.15, "STABLE",   "CONTINUE",           "Normal operation. No action required."),
    (0.30, "WATCH",    "INCREASE_SAMPLING",   "Minor distributional shift detected. Increase monitoring frequency and flag batch for review."),
    (0.50, "ALERT",    "RETRAIN_RECOMMENDED", "Significant drift detected. Retraining on fresh data is recommended. Notify data steward."),
    (1.01, "ROLLBACK", "ROLLBACK",            "Critical drift detected. Immediate rollback to last stable model checkpoint triggered."),
]


def classify_drift(drift_score):
    """
    Map a raw drift share (0.0 – 1.0) to a graduated response tier.
    Returns a dict with: status, action, description.
    """
    for max_val, status, action, description in DRIFT_TIERS:
        if drift_score <= max_val:
            return {
                "status":      status,
                "action":      action,
                "description": description,
            }
    # Fallback (should never reach here)
    return {"status": "ROLLBACK", "action": "ROLLBACK", "description": "Unknown drift level."}


def check_drift(reference_df, current_df, threshold=0.2):
    """
    Novelty 4: Detect data drift between reference (Site A) and current (Site B)
    using Evidently, and apply a GRADUATED RESPONSE rather than a binary panic switch.

    Tiers:
        STABLE   (drift < 0.15) → continue normal operation
        WATCH    (0.15 – 0.30)  → increase monitoring frequency
        ALERT    (0.30 – 0.50)  → retrain recommended
        ROLLBACK (> 0.50)       → immediate checkpoint fallback
    """
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    # Extract drift share — handle both old and new Evidently API
    result = report.as_dict()
    try:
        metrics_result = result["metrics"][0]["result"]
        if "share_of_drifted_columns" in metrics_result:
            drift_score = float(metrics_result["share_of_drifted_columns"])
        elif "dataset_drift" in metrics_result and isinstance(metrics_result["dataset_drift"], float):
            drift_score = metrics_result["dataset_drift"]
        else:
            drift_score = float(bool(metrics_result.get("dataset_drift", False)))
    except (KeyError, IndexError):
        drift_score = 0.5  # Conservative unknown default

    # Apply graduated response
    tier = classify_drift(drift_score)

    event = {
        "timestamp":   str(datetime.datetime.now()),
        "drift_score": round(drift_score, 4),
        "status":      tier["status"],
        "action":      tier["action"],
        "description": tier["description"],
    }

    os.makedirs("logs", exist_ok=True)
    with open("logs/audit_log.json", "a") as f:
        f.write(json.dumps(event) + "\n")

    # Console output
    icons = {"STABLE": "✅", "WATCH": "🟡", "ALERT": "🟠", "ROLLBACK": "🚨"}
    icon = icons.get(tier["status"], "❓")
    print(f"Drift Score: {drift_score:.2f} → {icon} {tier['status']} — {tier['action']}")
    if tier["action"] == "ROLLBACK":
        print("🚨 TRIGGERING AUTOMATED CHECKPOINT ROLLBACK")

    # Return full event + backward-compatible 'action' key for existing callers
    event["action"] = tier["action"]
    return event
