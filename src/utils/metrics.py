from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(p: Any) -> Dict[str, float]:
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return {
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "f1_class0": float(f1_score(labels, preds, pos_label=0, zero_division=0)),
        "f1_class1": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
    }
