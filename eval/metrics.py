"""
Automated metrics computed against the golden set:

- Intent classification: accuracy, per-class precision/recall/F1
- Decision (auto-handle vs escalate): accuracy, and specifically
  false-auto-handle rate — an escalate-worthy message that got auto-handled
  is the costly error direction, so we report it separately rather than
  letting it get buried in overall accuracy (see REPORT.md, "What is
  misleading about my headline number?").
"""

from collections import defaultdict


def classification_report(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    per_class = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = sum(1 for t in y_true if t == label)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}

    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
    return {"accuracy": accuracy, "per_class": per_class}


def decision_report(y_true: list[str], y_pred: list[str]) -> dict:
    """y_true/y_pred are 'auto_handle' / 'escalate' strings."""
    n = len(y_true)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n if n else 0.0

    # The costly error: gold says escalate, system auto-handled it anyway.
    escalate_true_idx = [i for i, t in enumerate(y_true) if t == "escalate"]
    false_auto_handles = sum(1 for i in escalate_true_idx if y_pred[i] == "auto_handle")
    false_auto_handle_rate = false_auto_handles / len(escalate_true_idx) if escalate_true_idx else 0.0

    # The other-direction cost: gold says auto_handle, system escalated
    # anyway (wastes human time, but far cheaper than the above).
    auto_true_idx = [i for i, t in enumerate(y_true) if t == "auto_handle"]
    over_escalations = sum(1 for i in auto_true_idx if y_pred[i] == "escalate")
    over_escalation_rate = over_escalations / len(auto_true_idx) if auto_true_idx else 0.0

    return {
        "accuracy": accuracy,
        "false_auto_handle_rate": false_auto_handle_rate,
        "false_auto_handle_count": false_auto_handles,
        "escalate_worthy_total": len(escalate_true_idx),
        "over_escalation_rate": over_escalation_rate,
        "over_escalation_count": over_escalations,
        "auto_worthy_total": len(auto_true_idx),
    }


def print_report(title: str, report: dict):
    print(f"\n--- {title} ---")
    for k, v in report.items():
        if k == "per_class":
            continue
        if isinstance(v, float):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")
    if "per_class" in report:
        print("per-class:")
        for label, m in report["per_class"].items():
            print(f"  {label:30s} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} n={m['support']}")
