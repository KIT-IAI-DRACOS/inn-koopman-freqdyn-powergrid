from __future__ import annotations

from ._base import BaseRegressor
from ._deephodmd import DeepHODMD
# from ._mulinndmd import MultiScaleINNDMD
from ._deephodmd import DeepHODMD

__all__ = [
    "DeepHODMD",
    "TwoStageDeepHODMD",
    "DeepHODMD",
    # "MultiScaleINNDMD",
]