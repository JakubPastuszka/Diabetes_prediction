"""Pipeline: preprocess → split → train → evaluate."""

from __future__ import annotations

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
