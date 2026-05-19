import torch
import pytest
from unittest.mock import MagicMock, patch

from src.utils.losses import FocalLoss, compute_class_weights
from src.utils.metrics import compute_metrics
import pandas as pd
import numpy as np


def test_focal_loss_no_gamma():
    loss_fn = FocalLoss()
    logits = torch.randn(4, 2)
    labels = torch.randint(0, 2, (4,))
    loss = loss_fn(logits, labels)
    assert loss.item() > 0


def test_focal_loss_with_gamma():
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.randn(4, 2)
    labels = torch.randint(0, 2, (4,))
    loss = loss_fn(logits, labels)
    assert loss.item() > 0


def test_focal_loss_with_class_weights():
    weights = torch.tensor([1.0, 2.0])
    loss_fn = FocalLoss(class_weights=weights)
    logits = torch.randn(4, 2)
    labels = torch.randint(0, 2, (4,))
    loss = loss_fn(logits, labels)
    assert loss.item() > 0


def test_compute_class_weights():
    df = pd.DataFrame({"label": [0, 0, 0, 1]})
    w = compute_class_weights(df, "label")
    assert w.shape == (2,)
    assert w[1] > w[0]


def test_compute_metrics():
    preds = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]])
    labels = np.array([0, 1, 0, 1])

    class P:
        predictions = preds
        label_ids = labels

    m = compute_metrics(P())
    assert "f1_macro" in m
    assert "accuracy" in m
    assert 0.0 <= m["f1_macro"] <= 1.0


def test_balanced_sampler():
    from src.data.dataset import create_balanced_sampler
    labels = [0] * 80 + [1] * 20
    sampler = create_balanced_sampler(labels, num_classes=2)
    assert sampler.num_samples == 40  # 20 * 2
