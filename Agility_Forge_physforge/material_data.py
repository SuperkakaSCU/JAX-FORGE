"""Temperature-dependent material data tables for 15-5PH stainless steel.

HU: 15-5PH -- 20~600 are based on ChatGPT, waiting for calibration.
"""

import jax.numpy as np


# ---------------------------------------------------------------------------
# Hardening saturation Q(T) interpolation data
# ---------------------------------------------------------------------------
Q_temps = np.array([20, 200, 300, 426, 538, 682, 800, 898, 1000, 1100])
Q_reducs = np.array([145., 110., 131., 219., 276, 250., 300., 220., 80., 80.]) / 145.0

# ---------------------------------------------------------------------------
# Hardening rate b(T) interpolation data
# ---------------------------------------------------------------------------
b_temps = np.array([20, 100, 200, 300, 400, 500, 682, 800., 898, 1000, 1100])
b_reducs = np.array([25., 20., 12., 8., 4., 4., 4.8, 4.7, 4.5, 4.3, 4.3]) / 25.
