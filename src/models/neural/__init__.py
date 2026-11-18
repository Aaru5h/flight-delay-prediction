"""Neural models for the DL milestone (Steps 6 and 7).

PyTorch is imported first so its OpenMP runtime initialises before any other
library's on macOS. The neural modules never import XGBoost.
"""

import torch  # noqa: F401
