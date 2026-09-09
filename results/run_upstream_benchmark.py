#!/usr/bin/env python3
"""Run benchmark_gsm8k.py against the upstream FlagGems clone."""

from __future__ import annotations

import runpy
import sys


SOURCE = "/home/swj/FlagGems_upstream/src"
sys.meta_path = [
    finder
    for finder in sys.meta_path
    if getattr(finder, "__module__", "") != "__editable___flag_gems_0_0_0_finder"
]
sys.path.insert(0, SOURCE)

sys.argv = [
    "/home/swj/Trident/benchmark_gsm8k.py",
    "--warmup-count",
    "132",
    "--normal-count",
    "1187",
    "--max-new-tokens",
    "128",
    "--log-interval",
    "64",
    "--result-file",
    "/home/swj/Trident/results/flag_gems_upstream_retry_results.json",
    "--incremental-log-file",
    "/home/swj/Trident/results/flag_gems_upstream_retry_incremental.jsonl",
]

runpy.run_path("/home/swj/Trident/benchmark_gsm8k.py", run_name="__main__")
