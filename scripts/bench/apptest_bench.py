"""Server-side cost of each Quick Plot interaction, via streamlit AppTest.

Usage: python apptest_bench.py <repo_root> <page_relpath>
Times one full script run per interaction (what Streamlit executes per click).
"""

import statistics
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
repo = Path(sys.argv[1]).resolve()
page = repo / sys.argv[2]
sys.path.insert(0, str(repo))

from streamlit.testing.v1 import AppTest  # noqa: E402

EX = repo / "eagleplot" / "example-data" / "Artificial"


def timed(at, label, results, action=None):
    t0 = time.perf_counter()
    (action(at) if action else at).run()
    dt = (time.perf_counter() - t0) * 1000
    assert not at.exception, f"{label}: {at.exception}"
    results.setdefault(label, []).append(dt)


def scenario(dataset_file, plot_category, plot_label, prefix, slider_key):
    at = AppTest.from_file(str(page), default_timeout=120)
    at.run()
    at.session_state["datasets"] = {dataset_file: pd.read_csv(EX / dataset_file)}
    at.session_state["current_dataset"] = dataset_file
    at.run()
    at.selectbox(key="qp_category").set_value(plot_category).run()
    at.selectbox(key="qp_plottype").set_value(plot_label).run()
    res = {}
    timed(at, "generate", res, lambda a: a.button(key="qp_generate").click())
    for i in range(6):
        timed(at, "edit title (plot on screen)", res,
              lambda a, i=i: a.text_input(key=f"{prefix}_title").set_value(f"Title {i}"))
    if slider_key:
        for v in (0.3, 0.5, 0.7, 0.4, 0.6, 0.8):
            timed(at, "move slider (plot on screen)", res,
                  lambda a, v=v: a.slider(key=slider_key).set_value(v))
    for _ in range(3):
        timed(at, "generate", res, lambda a: a.button(key="qp_generate").click())
    return res


def report(name, res):
    print(f"\n## {name}")
    for label, xs in res.items():
        print(f"  {label:32s} median {statistics.median(xs):7.0f} ms   (n={len(xs)}, max {max(xs):.0f})")


if __name__ == "__main__":
    report("Matplotlib histogram, normal_distribution.csv",
           scenario("normal_distribution.csv", "Matplotlib", "Histogram", "histogram", "histogram_alpha"))
    report("Matplotlib line plot, ir_spectrum.csv",
           scenario("ir_spectrum.csv", "Matplotlib", "Line Plot", "line_plot", None))
