"""Ensure the module-level FastAPI app is created in an explicit test environment."""
import os

os.environ.setdefault("RAGS_ENV", "test")
