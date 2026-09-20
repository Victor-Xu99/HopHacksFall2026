"""Persist a fitted HarmScoringModel so a restart does not retrain.

Cache key is (source, model_type). If SafetyHops is de-leaked or resampled,
call invalidate() or delete models/ before the next start.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "models"


def cache_path(source: str, model_type: str, directory: Optional[Path] = None) -> Path:
    safe = source.replace("/", "_").replace("\\", "_")
    return (directory or DEFAULT_CACHE_DIR) / f"{safe}_{model_type}.joblib"


def load_cached(source: str, model_type: str, directory: Optional[Path] = None):
    """Return a previously trained model, or None."""
    try:
        import joblib
    except ImportError:
        logger.debug("joblib is not installed; skipping model cache.")
        return None

    folder = directory or DEFAULT_CACHE_DIR
    path = cache_path(source, model_type, folder)
    legacy = folder / f"{source.replace('/', '_')}_{model_type}.pkl"
    target = path if path.exists() else legacy if legacy.exists() else None
    if target is None:
        return None
    try:
        model = joblib.load(target)
        logger.info("Loaded cached model for %s/%s from %s", source, model_type, target)
        return model
    except Exception as exc:  # noqa: BLE001 — a corrupt file should not crash ranking
        logger.warning("Could not load cached model (%s); will retrain.", exc)
        return None


def save_cached(source: str, model_type: str, model, directory: Optional[Path] = None) -> Optional[Path]:
    """Write the fitted model. Returns the path, or None if persistence failed."""
    try:
        import joblib
    except ImportError:
        logger.debug("joblib is not installed; skipping model cache.")
        return None

    folder = directory or DEFAULT_CACHE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = cache_path(source, model_type, folder)
    try:
        joblib.dump(model, path)
        logger.info("Saved model for %s/%s to %s", source, model_type, path)
        return path
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not persist model: %s", exc)
        return None


def invalidate(
    source: Optional[str] = None,
    model_type: Optional[str] = None,
    directory: Optional[Path] = None,
) -> int:
    """Delete cached files. Omit source and model_type to wipe the directory."""
    folder = directory or DEFAULT_CACHE_DIR
    if not folder.exists():
        return 0
    removed = 0
    if source and model_type:
        for path in (
            cache_path(source, model_type, folder),
            folder / f"{source.replace('/', '_')}_{model_type}.pkl",
        ):
            if path.exists():
                path.unlink()
                removed += 1
        return removed
    for path in list(folder.glob("*.joblib")) + list(folder.glob("*.pkl")):
        path.unlink()
        removed += 1
    return removed
