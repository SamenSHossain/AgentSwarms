import base64
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarmprov import plotting  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
# a valid 1x1 PNG, written in place of every figure outside the tests marked `figures`
ONE_PIXEL_PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
SAVE = plotting.save


def _one_pixel(fig, path) -> str:
    plotting.plt.close(fig)
    Path(path).write_bytes(ONE_PIXEL_PNG)
    return str(path)


@pytest.fixture(scope="session", autouse=True)
def _figures_not_rasterised():
    """Figures are drawn but not rasterised (the tight-bbox pass is most of a figure's cost): plotting.save writes a
    1-pixel PNG, which the report links and html inlines like any other.  Session-wide, so the shared runs are built
    this way too."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(plotting, "save", _one_pixel)
        yield


@pytest.fixture(autouse=True)
def _figures_rasterised(request, monkeypatch):
    """A test marked ``figures`` looks at the images it draws: plotting.save writes them."""
    if request.node.get_closest_marker("figures"):
        monkeypatch.setattr(plotting, "save", SAVE)


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


@pytest.fixture(scope="session")
def synth_strict(synth_run, tmp_path_factory):
    """A copy of the seed-2 run with its strict report built beside the merged one (no resamples)."""
    from swarmprov import report
    from swarmprov.schema import RunDir
    d = tmp_path_factory.mktemp("synth_strict") / "run"
    shutil.copytree(synth_run[0].path, d)
    run = RunDir(d)
    report.build(run, mapping="strict", boot=0)
    return run


@pytest.fixture(scope="session")
def synth_null(tmp_path_factory):
    """The 60-agent seed-2 swarm with copy_rate 0: answers are often public already, but nobody copies, so no answer
    is instant and A2's latency outcome has no variance.  Run through ``swarmprov run --gold`` on a label directory:
    ``toy_labels.jsonl``, four of whose eight labelled posts are in the run, ``heldout_labels.jsonl``, all of whose
    four are, and ``notes.jsonl``, which is no label file.  Returns (RunDir, truth dir)."""
    import json
    from swarmprov import cli, synth
    from swarmprov.schema import RunDir
    d = tmp_path_factory.mktemp("synth_null")
    path = synth.write(d / "transcript.jsonl", n_agents=60, seed=2, copy_rate=0.0)
    posts = [json.loads(line) for line in open(path)]
    path.write_text("".join(json.dumps(p | {"id": f"m{i}"}) + "\n" for i, p in enumerate(posts)))   # ids to label
    cfg = json.loads((d / "config.json").read_text()) | {"bootstrap_reps": 0}
    (d / "cfg.json").write_text(json.dumps(cfg))
    labels = d / "labels"
    labels.mkdir()
    for split, ids in (("toy", ("m0", "m1", "m2", "m3", "x0", "x1", "x2", "x3")), ("heldout", ("m4", "m5", "m6", "m7"))):
        rows = [{"event_id": eid, "split": split, "labeller": "toy",
                 "gold": {"is_answer": True, "labelled": True, "answers": []}} for eid in ids]
        (labels / f"{split}_labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (labels / "notes.jsonl").write_text(path.read_text())
    assert cli.main(["run", str(path), "-o", str(d / "run"), "--adapter", "chat", "--config", str(d / "cfg.json"),
                     "--gold", str(labels)]) == 0
    return RunDir(d / "run"), d


@pytest.fixture(scope="session")
def wiki_run(tmp_path_factory):
    """The full wiki dump ingested and extracted once (about 25 s), for the tests on the real data."""
    if not (RAW / "events.jsonl").exists():
        pytest.skip("full wiki dump not present")
    from swarmprov import pipeline
    from swarmprov.schema import RunDir
    run = RunDir(tmp_path_factory.mktemp("wiki") / "run")
    pipeline.ingest(RAW, run)
    pipeline.extract(run)
    return run
