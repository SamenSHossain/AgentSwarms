import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def synth_run(tmp_path_factory):
    """The synthetic swarm run once for every module that checks it: (RunDir, truth table)."""
    from swarmprov import pipeline, synth
    d = tmp_path_factory.mktemp("synth")
    path = synth.write(d / "transcript.jsonl", n_agents=60, copy_rate=0.7, seed=2)
    run = pipeline.run_all(path, d / "run", adapter="chat", config=str(d / "config.json"))
    return run, pd.read_csv(d / "truth.csv")


@pytest.fixture(scope="session")
def synth_first(tmp_path_factory):
    """The seed-2 swarm again, but copying agents take the *first* post on the board (no report stage)."""
    from swarmprov import pipeline, synth
    from swarmprov.schema import RunDir
    d = tmp_path_factory.mktemp("synth_first")
    path = synth.write(d / "transcript.jsonl", n_agents=60, copy_rate=0.7, seed=2, copy_from="first")
    run = RunDir(d / "run")
    pipeline.ingest(path, run, "chat", str(d / "config.json"))
    pipeline.extract(run)
    pipeline.expose(run)
    pipeline.build_graph(run)
    return run, pd.read_csv(d / "truth.csv")
