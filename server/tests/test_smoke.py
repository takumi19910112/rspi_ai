"""Smoke tests: ensure every app module imports cleanly."""

import importlib

import pytest

MODULES = [
    "app",
    "app.config",
    "app.llm",
    "app.main",
    "app.pipeline",
    "app.stt",
    "app.tts",
    "app.ws_protocol",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)
