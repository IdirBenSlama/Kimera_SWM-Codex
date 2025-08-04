"""backend.core.knowledge_base
--------------------------------
Simple filesystem–based knowledge-base interface used by monitoring routes.

The *real* semantic vault implementation is not yet finalised.  To keep the API
operational – and to honour the Zero-Debugging constraint – we provide a
light-weight placeholder that lists sub-directories inside ``knowledge_base/``
and attempts to read an optional ``metadata.json`` file per vault.

Once the definitive vault architecture is implemented this module can be
expanded or replaced transparently because the public surface is deliberately
minimal (``KNOWLEDGE_BASE_DIR``, ``list_vaults``, ``get_vault_metadata``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------

# Locate or create the root directory for knowledge vaults.
KNOWLEDGE_BASE_DIR: Path = Path.cwd() / "knowledge_base"
KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
logger.debug("Knowledge-base directory set to %s", KNOWLEDGE_BASE_DIR)

# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------


def list_vaults() -> List[str]:  # noqa: D401
    """Return the names of all vault directories inside :pydata:`KNOWLEDGE_BASE_DIR`."""
    try:
        vaults = [p.name for p in KNOWLEDGE_BASE_DIR.iterdir() if p.is_dir()]
        logger.debug("Discovered %d vault(s): %s", len(vaults), vaults)
        return vaults
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to enumerate vaults: %s", exc)
        return []


def get_vault_metadata(vault_name: str) -> Dict[str, Any]:  # noqa: D401
    """Return metadata for *vault_name* if a ``metadata.json`` file is present."""
    vault_dir = KNOWLEDGE_BASE_DIR / vault_name
    metadata_file = vault_dir / "metadata.json"

    if not vault_dir.exists():
        logger.warning("Requested metadata for non-existent vault '%s'", vault_name)
        return {"name": vault_name, "error": "Vault not found"}

    if not metadata_file.exists():
        logger.info("No metadata.json found for vault '%s'", vault_name)
        return {"name": vault_name, "description": "Metadata not available"}

    try:
        with metadata_file.open("r", encoding="utf-8") as fp:
            data: Dict[str, Any] = json.load(fp)
        logger.debug("Loaded metadata for vault '%s'", vault_name)
        return data
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to read metadata for vault '%s': %s", vault_name, exc)
        return {"name": vault_name, "error": str(exc)}
