"""CompanyBrain Engineer harness as a library.

`app/engineer/api.py` is the product surface any wrapper should target. The other
modules are its implementation and are free to change shape.

The package expects `app/` on `sys.path` (it imports the sibling `api_clients`,
`model_registry`, `repository_retrieval`, `engineer_grounding`, `resilient_calls`
modules directly), so a wrapper does:

    import sys; sys.path.insert(0, "app")
    from engineer import api
"""

from __future__ import annotations

import sys
from pathlib import Path


_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

__all__ = ["api"]
