import math
from typing import Optional

import torch
import torch.nn as nn

# v12: distribution utilities centralized in src.utils.distribution. Re-export here
# for backward compat with code that does `from src.utils.losses import N_BINS, ...`.
from src.utils.distribution import (  # noqa: F401
    N_BINS,
    BIN_WIDTH,
    _TEST_PMF,
    compute_target_weights,
    empirical_pmf_y as compute_empirical_pmf,
    empirical_pmf_joint as compute_empirical_pmf_cell,
)

__all__ = [
    "N_BINS", "BIN_WIDTH", "_TEST_PMF",
    "compute_target_weights", "compute_empirical_pmf", "compute_empirical_pmf_cell",
    "WeightedMSELoss", "sliced_wasserstein",
]


class WeightedMSELoss(nn.Module):
    """Challenge metric formula with adaptive Lagrangian λ on the fairness gap.

    Per-group weighted MSE:
        Err_g = Σ w_i·(p-y)² / Σ w_i   with w_i = 1/30 + y_i  (+ optional sample reweight)

    Training loss (Lagrangian):
        L = (Err_F + Err_M)/2 + λ_adapt · |Err_F - Err_M|

    v17.3: log-ratio update from CLEAN val signal (eval_err_diff over 15k samples),
    once per epoch. Signal moteur = log(val_err_diff / threshold), invariant multiplicatif
    → symétrique en ascent/descent. Pas de cap explicite : log() croît sub-linéairement,
    soft-saturation naturelle (worst observé en DB : Δλ ≈ 0.6 pour spike val=0.011).

    Update rule:
        Δλ      = lambda_lr · log(max(val_err_diff, ε) / threshold)
        λ_{t+1} = clip(λ_t + Δλ, lambda_min, lambda_max)

    Properties:
      - val_err = thr    ⇒ Δλ = 0                              (équilibre)
      - val_err = thr·e  ⇒ Δλ = +lr                            (ex: +0.2)
      - val_err = thr/e  ⇒ Δλ = -lr                            (symétrique)
      - val_err → 0      ⇒ Δλ → -∞ (clip via lambda_min)
      - val_err = 10·thr ⇒ Δλ = +lr·log(10) ≈ +2.3·lr          (=+0.46 si lr=0.2)

    `lambda_min=1.0` = poids métrique : training pousse fairness AU MOINS autant que
    la métrique même quand val_err < threshold (= pas de relâchement sous métrique).
    `lambda_max=3.0` : marge haute, sat naturelle du log suffit (pas de jump style v17).
    Challenge metric (compute_score) toujours évalué avec λ_metric = 1.0.
    """

    def __init__(
        self,
        weight_offset: float = 1.0 / 30.0,
        focal_gamma: float = 0.0,
        lambda_init: float = 1.0,
        lambda_lr: float = 0.2,
        lambda_max: float = 3.0,
        lambda_min: float = 1.0,
        lambda_threshold: float = 0.0005,
    ) -> None:
        super().__init__()
        self.weight_offset = weight_offset
        self.focal_gamma = focal_gamma
        self.lambda_lr = float(lambda_lr)
        self.lambda_max = float(lambda_max)
        self.lambda_min = float(lambda_min)
        self.lambda_threshold = float(lambda_threshold)
        self.register_buffer("lambda_adapt", torch.tensor(float(lambda_init)))

    def update_lambda(self, val_err_diff: float) -> float:
        """Called once per epoch by LambdaLogCallback after val eval. log-ratio update,
        symétrique multiplicative autour de threshold. Log sature naturellement.
        """
        with torch.no_grad():
            err = max(float(val_err_diff), 1e-9)
            delta = self.lambda_lr * math.log(err / max(self.lambda_threshold, 1e-9))
            new_lambda = float(self.lambda_adapt.item()) + delta
            new_lambda = max(self.lambda_min, min(self.lambda_max, new_lambda))
            self.lambda_adapt.fill_(new_lambda)
        return new_lambda

    def forward(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        sample_loss_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        preds = preds.view(-1)
        if labels.dim() == 2 and labels.size(1) >= 2:
            targets = labels[:, 0]
            gender = labels[:, 1]
        else:
            targets = labels.view(-1)
            gender = None

        err = (preds - targets) ** 2
        w = self.weight_offset + targets

        if sample_loss_weight is not None:
            w = w * sample_loss_weight.to(w.device).to(w.dtype)

        if self.focal_gamma > 0:
            # v26 fix: canonical absolute focal loss (was squared-err + 0.05 offset
            # in v19). Pivot at 10% error gives weight 1.0; larger errors are
            # amplified, smaller ones attenuated. See Antoine's v26 commit.
            diff = (preds - targets).abs()
            focal_weight = (diff.detach() / 0.1).pow(self.focal_gamma)
            w = w * focal_weight

        if gender is None:
            return (w * err).sum() / w.sum().clamp(min=1e-8)

        mask_f = gender < 0.5
        mask_m = gender >= 0.5

        if not (mask_f.any() and mask_m.any()):
            return (w * err).sum() / w.sum().clamp(min=1e-8)

        err_f = (w[mask_f] * err[mask_f]).sum() / w[mask_f].sum().clamp(min=1e-8)
        err_m = (w[mask_m] * err[mask_m]).sum() / w[mask_m].sum().clamp(min=1e-8)
        err_diff = (err_f - err_m).abs()

        # λ used in training loss is the externally-updated buffer; eval uses λ_metric=1.
        lam = self.lambda_adapt if self.training else torch.tensor(1.0, device=err_diff.device, dtype=err_diff.dtype)
        return (err_f + err_m) / 2.0 + lam * err_diff


def sliced_wasserstein(x: torch.Tensor, y: torch.Tensor, n_projections: int = 100) -> torch.Tensor:
    """Sliced Wasserstein-2 distance between two feature sets (v16: L=100).

    Projects x, y onto random 1D directions, computes 1D OT (= sorted L2) per
    projection, averages. Differentiable, batch-size-agnostic (handles unequal
    sizes via linear interpolation on the empirical CDFs).
    """
    if x.shape[0] < 2 or y.shape[0] < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)

    d = x.shape[-1]
    proj = torch.randn(d, n_projections, device=x.device, dtype=x.dtype)
    proj = proj / proj.norm(dim=0, keepdim=True).clamp(min=1e-9)

    x_proj = (x @ proj).t()
    y_proj = (y @ proj).t()
    x_sorted, _ = x_proj.sort(dim=-1)
    y_sorted, _ = y_proj.sort(dim=-1)

    if x.shape[0] != y.shape[0]:
        n = max(x.shape[0], y.shape[0])
        x_sorted = torch.nn.functional.interpolate(x_sorted.unsqueeze(1), size=n, mode="linear", align_corners=True).squeeze(1)
        y_sorted = torch.nn.functional.interpolate(y_sorted.unsqueeze(1), size=n, mode="linear", align_corners=True).squeeze(1)

    return ((x_sorted - y_sorted) ** 2).mean()


def sinkhorn_distance(
    x: torch.Tensor, y: torch.Tensor,
    eps: float = 0.1, n_iter: int = 50,
) -> torch.Tensor:
    """Entropic OT distance (Sinkhorn algorithm, Cuturi 2013) — log-space stable.

    Computes <P, C> where P is the optimal entropy-regularized transport plan
    between empirical distributions of x and y:
        P = argmin_{P∈U(a,b)} <P, C> - eps · H(P)
        U(a, b) = {P ≥ 0 : P·1 = a, P^T·1 = b}    (couplings with uniform marginals)
        C_ij = ||x_i - y_j||²                       (squared Euclidean cost)

    Plus précis que sliced_wasserstein (vraie OT régularisée, pas projections 1D),
    mais ~3-5× plus coûteux à batch=128. Numérique stable via log-sum-exp.

    Args:
        eps: entropy regularization. Plus petit = plus proche de W2 exact mais moins stable.
        n_iter: itérations Sinkhorn (converge en 20-100 typiquement)

    Returns:
        Coût d'OT régularisé approximation de W2² (scalar tensor).
    """
    import math
    if x.shape[0] < 2 or y.shape[0] < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)

    n_x, n_y = x.shape[0], y.shape[0]
    C = torch.cdist(x, y).pow(2)                                                  # (n_x, n_y)

    # Log-space marginals (uniform): log a_i = -log(n_x), log b_j = -log(n_y)
    log_a = torch.full((n_x,), -math.log(n_x), device=x.device, dtype=x.dtype)
    log_b = torch.full((n_y,), -math.log(n_y), device=y.device, dtype=y.dtype)
    log_K = -C / eps                                                              # log Gibbs kernel

    # Sinkhorn log-iterations:
    #   log_v = log_b - logsumexp(log_K + log_u[:, None], dim=0)
    #   log_u = log_a - logsumexp(log_K + log_v[None, :], dim=1)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(n_iter):
        log_v = log_b - torch.logsumexp(log_K + log_u[:, None], dim=0)
        log_u = log_a - torch.logsumexp(log_K + log_v[None, :], dim=1)

    # Transport plan in log-space: log P_ij = log u_i + log K_ij + log v_j
    log_P = log_u[:, None] + log_K + log_v[None, :]
    P = log_P.exp()
    return (P * C).sum()


