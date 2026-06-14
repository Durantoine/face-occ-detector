import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Constants from src/utils/distribution.py
N_BINS = 200
CENTERS = 0.5 * (np.linspace(0.0, 1.0, N_BINS + 1)[:-1] + np.linspace(0.0, 1.0, N_BINS + 1)[1:])

_TEST_PMF_05 = np.array([
    0.023880, 0.021935, 0.018046, 0.018046, 0.014993, 0.013726, 0.011194, 0.011194, 0.008277, 0.011940,
    0.019267, 0.019267, 0.016282, 0.015671, 0.014450, 0.014450, 0.013704, 0.013161, 0.012618, 0.012618,
    0.017096, 0.016926, 0.016757, 0.016757, 0.015332, 0.014823, 0.014314, 0.014314, 0.012822, 0.015468,
    0.018113, 0.018113, 0.016757, 0.015909, 0.015061, 0.014608, 0.013704, 0.013229, 0.012754, 0.014269,
    0.017299, 0.016485, 0.015671, 0.015694, 0.015739, 0.015264, 0.014789, 0.014156, 0.012890, 0.013839,
    0.014789, 0.013941, 0.013093, 0.012890, 0.012686, 0.012449, 0.012211, 0.011601, 0.010990, 0.010651,
    0.010312, 0.009362, 0.008412, 0.007734, 0.007055, 0.006682, 0.006309, 0.006038, 0.005766, 0.005178,
    0.004885, 0.004613, 0.004342, 0.003347, 0.002849, 0.002578, 0.002307, 0.002035, 0.011900, 0.001492,
    0.001085, 0.000950, 0.000882, 0.000814, 0.000746, 0.000339, 0.000339, 0.000373, 0.000407, 0.000339,
    0.000339, 0.000237, 0.000136, 0.000136, 0.000136, 0.000170, 0.000204, 0.000068, 0.000068, 0.000068,
], dtype=np.float64)
_TEST_PMF_05[78] = 0.001900; _TEST_PMF_05[92] = 0.000136
P_TEST_PMF = np.zeros(N_BINS, dtype=np.float64); P_TEST_PMF[:100] = _TEST_PMF_05
P_TEST_PMF = P_TEST_PMF / P_TEST_PMF.sum()

# Load ConvNext predictions
df = pd.read_csv('test_predictions_convnext.csv')
y_pred = df['FaceOcclusion'].values

# Compute density of predictions
# We use more bins for a smoother curve or a KDE
hist, bin_edges = np.histogram(y_pred, bins=N_BINS, range=(0, 1), density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(CENTERS, P_TEST_PMF * N_BINS, label='Target P_test (Challenge)', linewidth=2, color='black', linestyle='--')
plt.plot(bin_centers, hist, label='ConvNext Predictions', linewidth=2, color='blue')

plt.fill_between(bin_centers, hist, alpha=0.2, color='blue')
plt.title('Distribution Comparison: ConvNext Predictions vs P_test')
plt.xlabel('FaceOcclusion')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(0, 0.6) # P_test only goes up to ~0.5

# Save plot
plt.savefig('dist_comparison_convnext.png', dpi=150)
print("Plot saved to dist_comparison_convnext.png")

# Also print some stats
print(f"Prediction Mean: {y_pred.mean():.4f}")
print(f"Target P_test Mean: {np.sum(CENTERS * P_TEST_PMF):.4f}")
