"""
Backend package initializer.
This file makes `backend` a Python package so relative imports work
when running the app via `uvicorn backend.main:app`.
"""

__all__ = ["main", "database"]
