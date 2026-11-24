"""Smoke test: the Streamlit app renders and generates without GPT-2 weights."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_renders_and_generates():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "% copied" in labels
    at.sidebar.slider[0].set_value(3)
    at.sidebar.button[0].click().run()
    assert not at.exception
