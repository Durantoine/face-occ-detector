import numpy as np
import torch
import pytest

from src.utils.losses import ChallengeLoss
from src.utils.metrics import challenge_score
from src.utils.distribution import stratified_is_score, get_joint_weight_fn

def test_challenge_score_basic():
    preds = np.array([0.1, 0.2, 0.3, 0.4])
    gt = np.array([0.1, 0.2, 0.3, 0.4])
    gender = np.array([0.0, 1.0, 0.0, 1.0])
    m = challenge_score(preds, gt, gender)
    assert m["challenge_score"] < 1e-8
    assert m["err_diff"] < 1e-8

def test_challenge_score_disparity():
    # Female has higher error
    preds = np.array([0.5, 0.5, 0.0, 0.0])
    gt = np.array([0.0, 0.0, 0.0, 0.0])
    gender = np.array([0.0, 0.0, 1.0, 1.0])
    m = challenge_score(preds, gt, gender)
    assert m["err_F"] > m["err_M"]
    assert m["err_diff"] > 0
    # Score = (err_F + err_M)/2 + |err_F - err_M|
    expected = (m["err_F"] + m["err_M"]) / 2 + m["err_diff"]
    assert abs(m["challenge_score"] - expected) < 1e-8

def test_challenge_loss_forward():
    loss_fn = ChallengeLoss(lambda_gap=1.0)
    preds = torch.tensor([0.1, 0.3, 0.5, 0.7])
    labels = torch.tensor([0.1, 0.3, 0.5, 0.7])
    gender = torch.tensor([0.0, 0.0, 1.0, 1.0])
    loss = loss_fn(preds, labels, gender)
    assert loss.item() < 1e-6

def test_challenge_loss_with_gap():
    loss_fn = ChallengeLoss(lambda_gap=2.0)
    # F error is large, M error is 0
    preds = torch.tensor([1.0, 1.0, 0.0, 0.0])
    labels = torch.tensor([0.0, 0.0, 0.0, 0.0])
    gender = torch.tensor([0.0, 0.0, 1.0, 1.0])
    
    # Manually calculate expected
    # w_i = 1/30 + 0 = 1/30
    # err_f = (1/30 * 1^2 + 1/30 * 1^2) / (1/30 + 1/30) = 1.0
    # err_m = (1/30 * 0^2 + 1/30 * 0^2) / (1/30 + 1/30) = 0.0
    # loss = (1+0)/2 + 2.0 * |1-0| = 0.5 + 2.0 = 2.5
    
    loss, gap = loss_fn(preds, labels, gender, return_gap=True)
    assert abs(loss.item() - 2.5) < 1e-6
    assert abs(gap - 1.0) < 1e-6

def test_stratified_is_score_equivalence():
    preds = np.array([0.1, 0.2, 0.3, 0.4])
    gt = np.array([0.1, 0.2, 0.3, 0.4])
    gender = np.array([0.0, 1.0, 0.0, 1.0])
    
    # Without weights, should match challenge_score
    m1 = challenge_score(preds, gt, gender)
    m2 = stratified_is_score(preds, gt, gender, weight_fn=None)
    
    assert abs(m1["challenge_score"] - m2["challenge_score"]) < 1e-8

def test_weight_fn_regularized_ratio():
    from src.utils.distribution import W_FLOOR, W_CEIL, W_HI_FLOOR
    y_train = np.random.uniform(0, 0.5, 100)
    g_train = np.random.randint(0, 2, 100).astype(float)

    weight_fn = get_joint_weight_fn(y_train, g_train, target="hc", lam=0.10)
    weights = weight_fn(y_train, g_train)

    assert len(weights) == 100
    assert (weights >= W_FLOOR - 1e-6).all() and (weights <= W_CEIL + 1e-6).all()
    # Past Y_LOW occ no variant may fall below 1 -> kept high-occ floors UP to W_HI_FLOOR, never ignored
    assert np.allclose(weight_fn(np.array([0.6, 0.9]), np.array([0.0, 1.0])), W_HI_FLOOR, atol=1e-6)

def test_challenge_loss_adaptive_lambda():
    loss_fn = ChallengeLoss(lambda_gap=1.0, adaptive=True, lam_min=1.0, lam_max=5.0, threshold=0.001)
    
    # Initial lambda is 1.0
    assert loss_fn.lam == 1.0
    
    # Large gap -> lambda should increase
    new_lam = loss_fn.update_lambda(0.1) # current_gap = 0.1
    assert new_lam > 1.0
    
    # Small gap -> lambda should stay or decrease (but bounded by lam_min)
    new_lam = loss_fn.update_lambda(0.0001)
    assert new_lam == 1.0
