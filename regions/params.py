"""
Set global parameters for region extraction.

ARY 2024-Nov
"""
import numpy as np

# Grid resolution:
ns = 180
nph = 360

# Latitude above which to zero maps:
max_lat=np.deg2rad(50)

# Required threshold in br maps:
br_min = 50

# Maximum flux imbalance:
max_unbalance = 0.5

# Minimum size of a plage (pixels):
npix_min = 50

# Parameters for MWO spot lag correction:
mwo_lag_window_min = -6
mwo_lag_window_max = 0
mwo_lag_degree = 1
mwo_lag_width = 7

# Parameters for sunspot polarity map:
mwo_spot_polarity_alpha = 6
mwo_spot_polarity_k = 5

# Minimum fraction of pixels with polarity determined to define a "complete" region:
mwo_spot_complete_minfrac = 0.5

# Probability of flipping polarities in incomplete regions [to recover correct anti-Hale percentage]
incomplete_flip_threshold = 0.06

# Scaling factor to reduce flux of incomplete regions [to avoid overestimating axial dipole]:
incomplete_flux_scaling = 1.3
