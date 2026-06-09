"""Headless smoke test for the Streamlit app using AppTest.

Renders the app, switches to the pipeline page, clicks Run, and asserts no
exceptions occurred. Not part of the pytest suite (it needs streamlit), but
runnable on demand: `python ui/_smoke_test.py`.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent / "app.py")


def run():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, f"home page raised: {at.exception}"
    print("home: OK")

    # exercise the Run Pipeline page
    at.sidebar.radio[0].set_value("🚀 Run Pipeline").run()
    assert not at.exception, f"pipeline page raised: {at.exception}"
    # click the primary "Run pipeline" button
    run_btn = [b for b in at.button if "Run pipeline" in b.label]
    assert run_btn, "Run pipeline button not found"
    run_btn[0].click().run()
    assert not at.exception, f"pipeline run raised: {at.exception}"
    print("pipeline run: OK")

    # exercise Dedup Explorer
    at.sidebar.radio[0].set_value("🔁 Dedup Explorer").run()
    assert not at.exception, f"dedup page raised: {at.exception}"
    db = [b for b in at.button if "Run dedup" in b.label]
    if db:
        db[0].click().run()
        assert not at.exception, f"dedup run raised: {at.exception}"
    print("dedup: OK")

    # exercise Annotation QA
    at.sidebar.radio[0].set_value("🧑‍🤝‍🧑 Annotation QA").run()
    ab = [b for b in at.button if "agreement" in b.label.lower()]
    if ab:
        ab[0].click().run()
        assert not at.exception, f"annotation run raised: {at.exception}"
    print("annotation: OK")

    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    run()
