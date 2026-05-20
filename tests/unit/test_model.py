import numpy as np
import torch

from src.data.dataset import create_balanced_sampler
from src.inference.calibration import apply_bias, find_optimal_bias
from src.utils.losses import (
    GroupDROLoss,
    WeightedMSELoss,
    make_sampler_keys,
    occlusion_bucket,
    stratify_key,
)
from src.utils.metrics import compute_metrics, compute_score


def test_weighted_mse_basic():
    loss_fn = WeightedMSELoss()
    preds = torch.tensor([0.1, 0.3, 0.5])
    labels = torch.tensor([[0.0, 0.0], [0.3, 1.0], [0.5, 0.0]])
    assert loss_fn(preds, labels).item() >= 0


def test_weighted_mse_perfect_pred():
    loss_fn = WeightedMSELoss()
    preds = torch.tensor([0.1, 0.3, 0.5])
    labels = torch.tensor([[0.1, 0.0], [0.3, 1.0], [0.5, 0.0]])
    assert loss_fn(preds, labels).item() < 1e-6


def test_weighted_mse_with_focal():
    loss_fn = WeightedMSELoss(focal_gamma=1.0)
    preds = torch.tensor([0.0, 0.0])
    labels = torch.tensor([[0.5, 0.0], [0.1, 1.0]])
    assert loss_fn(preds, labels).item() > 0


def test_weighted_mse_with_fairness():
    loss_fn = WeightedMSELoss(fairness_lambda=1.0)
    preds = torch.tensor([0.5, 0.5, 0.0, 0.0])
    labels = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.5, 1.0], [0.5, 1.0]])
    assert loss_fn(preds, labels).item() > 0


def test_group_dro_loss():
    loss_fn = GroupDROLoss(alpha=1.0)
    preds = torch.tensor([0.5, 0.5, 0.0, 0.0])
    labels = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    out = loss_fn(preds, labels)
    assert out.item() > 0


def test_group_dro_worst_case_focus():
    fn_mean = GroupDROLoss(alpha=0.0)
    fn_worst = GroupDROLoss(alpha=1.0)
    preds = torch.tensor([0.0, 0.0, 0.5, 0.5])
    labels = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    assert fn_worst(preds, labels).item() >= fn_mean(preds, labels).item()


def test_compute_score_balanced():
    preds = np.array([0.1, 0.2, 0.3, 0.4])
    gt = np.array([0.1, 0.2, 0.3, 0.4])
    gender = np.array([0.0, 1.0, 0.0, 1.0])
    m = compute_score(preds, gt, gender)
    assert m["score"] < 1e-8


def test_compute_score_disparity():
    preds = np.array([0.5, 0.5, 0.0, 0.0])
    gt = np.array([0.0, 0.0, 0.0, 0.0])
    gender = np.array([0.0, 0.0, 1.0, 1.0])
    m = compute_score(preds, gt, gender)
    assert m["err_diff"] > 0
    assert m["score"] >= m["err_F"] / 2 + m["err_M"] / 2


def test_compute_metrics_hf_format():
    class P:
        predictions = np.array([0.1, 0.3])
        label_ids = np.array([[0.1, 0.0], [0.3, 1.0]])
    m = compute_metrics(P())
    assert "err_F" in m and "err_M" in m


def test_make_sampler_keys_strategies():
    import pandas as pd
    df = pd.DataFrame({
        "gender": [0, 0, 1, 1, 1],
        "FaceOcclusion": [0.0, 0.1, 0.2, 0.3, 0.9],
    })
    for s in ("gender", "occlusion", "gender_x_occ"):
        keys = make_sampler_keys(df, strategy=s, n_buckets=3)
        assert len(keys) == 5
        assert keys.max() >= 0


def test_balanced_sampler():
    gender = [0] * 80 + [1] * 20
    sampler = create_balanced_sampler(gender, num_groups=2)
    assert sampler.num_samples == 40


def test_bias_correction_improves_score():
    rng = np.random.default_rng(0)
    n = 200
    gt = rng.uniform(0, 0.5, n)
    gender = rng.integers(0, 2, n).astype(float)
    preds = gt + np.where(gender < 0.5, 0.05, -0.03) + rng.normal(0, 0.02, n)
    preds = np.clip(preds, 0, 1)
    before = compute_score(preds, gt, gender)
    df, dm, _ = find_optimal_bias(preds, gt, gender)
    after = compute_score(apply_bias(preds, gender, df, dm), gt, gender)
    assert after["score"] <= before["score"]


def test_occlusion_bucket_and_stratify():
    occ = np.array([0.0, 0.1, 0.2, 0.5, 0.9])
    g = np.array([0, 0, 1, 1, 0])
    keys = stratify_key(g, occ, n_buckets=3)
    assert len(set(keys.tolist())) >= 2
