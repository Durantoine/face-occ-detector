from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as _beta


@dataclass(frozen=True)
class MixSpikeBeta:
    """P(Y) = p0 · δ(0) + (1 − p0) · Beta(α, β; 2Y)

    Y vit sur [0, 0.5] (FaceOcclusion bornée), mappé sur [0, 1] via 2Y pour utiliser
    Beta standard. Le terme spike capture les "vrais no-occlusion" qui ne suivent pas
    la queue continue (utile pour P_test où ~7% des images sont parfaitement claires).

    Pour P_train, p0=0 marche : la "spike-like" mass au bin 0 est entièrement absorbée
    par une Beta gauche-asymétrique (α<1).
    """
    p0: float
    alpha: float
    beta: float

    def pmf_at_bins(self, n_bins: int = 20, bin_width: float = 0.025) -> np.ndarray:
        edges = np.linspace(0.0, n_bins * bin_width, n_bins + 1)
        beta_cdf_edges = _beta.cdf(edges * 2.0, self.alpha, self.beta)
        pmf_beta = np.diff(beta_cdf_edges)
        pmf = (1.0 - self.p0) * pmf_beta
        pmf[0] += self.p0
        return pmf

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng()
        is_spike = rng.random(n) < self.p0
        beta_y = _beta.rvs(self.alpha, self.beta, size=n, random_state=rng) / 2.0
        return np.where(is_spike, 0.0, beta_y)


# P_test : extrait du PDF page 3 (29980 images, IDEMIA test set) + fit Mix(spike + Beta)
# par minimisation L2 (cf docstring losses.py historique, L2=0.0334). 7% de "vrais
# no-occlusion" + Beta(1.67, 2.85) skewed-right. Le spike est nécessaire pour capturer
# la population qualitativement distincte des visages parfaitement clairs.
_TEST_DIST = MixSpikeBeta(p0=0.069, alpha=1.67, beta=2.85)

# P_train : fit Mix(spike + Beta) sur train.csv (100k subset local) via Nelder-Mead.
# p0=0 trouvé optimal → pas de vrai spike au bin 0, la mass est absorbée par
# Beta(0.65, 3.43) avec α<1 (densité divergente à 0+). L2=0.000493.
# Cf src/utils/distributions.py:fit_train_distribution pour reproduire le fit.
_TRAIN_DIST = MixSpikeBeta(p0=0.0, alpha=0.647, beta=3.431)


def fit_train_distribution(train_csv: str = "data/raw/train.csv",
                            label_col: str = "FaceOcclusion",
                            n_bins: int = 20,
                            bin_width: float = 0.025) -> MixSpikeBeta:
    """Refit P_train depuis le CSV. Utilitaire pour reproduire / vérifier _TRAIN_DIST."""
    import pandas as pd
    from scipy.optimize import minimize

    df = pd.read_csv(train_csv)
    y = df[label_col].astype(float).values
    edges = np.linspace(0, n_bins * bin_width, n_bins + 1)
    y_clip = np.clip(y, 0.0, edges[-1] - 1e-9)
    hist, _ = np.histogram(y_clip, bins=edges)
    target = hist / hist.sum()

    def loss(params: np.ndarray) -> float:
        p0, a, b = params
        if not (0.0 <= p0 < 1.0 and a > 0 and b > 0):
            return 1e6
        pmf_beta = np.diff(_beta.cdf(edges * 2.0, a, b))
        pmf = (1.0 - p0) * pmf_beta
        pmf[0] += p0
        return float(((pmf - target) ** 2).sum())

    best, best_loss = None, np.inf
    for p0_init in [0.0, 0.1, 0.2, 0.3]:
        for a_init in [0.5, 1.0, 1.5, 2.0]:
            for b_init in [3.0, 5.0, 8.0]:
                res = minimize(loss, [p0_init, a_init, b_init],
                               method="Nelder-Mead", options={"xatol": 1e-5})
                if res.fun < best_loss:
                    best_loss, best = res.fun, res.x
    return MixSpikeBeta(p0=float(best[0]), alpha=float(best[1]), beta=float(best[2]))
