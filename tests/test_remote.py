import gzip
import json
from pathlib import Path

import pytest

from swarmprov import pipeline, remote
from swarmprov.adapters.chat import ChatAdapter
from swarmprov.schema import RunDir

huggingface_hub = pytest.importorskip("huggingface_hub")


def test_local_paths_pass_through(tmp_path, monkeypatch):
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kw: pytest.fail("downloaded a local path"))
    assert remote.resolve(tmp_path / "a.json") == tmp_path / "a.json"
    assert remote.resolve(str(tmp_path)) == tmp_path
    assert not remote.is_remote(tmp_path)


def test_file_uri_downloads_one_file(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kw: calls.append(kw) or str(tmp_path / kw["filename"]))
    out = remote.resolve("hf://datasets/aidigestorg/ai-village/chat_messages.jsonl.gz")
    assert out == tmp_path / "chat_messages.jsonl.gz"
    assert calls == [{"repo_id": "aidigestorg/ai-village", "repo_type": "dataset", "revision": None,
                      "cache_dir": None, "filename": "chat_messages.jsonl.gz"}]


def test_repo_uri_downloads_snapshot_with_revision(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: calls.append(kw) or str(tmp_path))
    assert remote.resolve("hf://datasets/aidigestorg/ai-village@v2") == tmp_path
    assert calls == [{"repo_id": "aidigestorg/ai-village", "repo_type": "dataset", "revision": "v2", "cache_dir": None}]
    calls.clear()
    remote.resolve("hf://owner/model")
    assert calls[0]["repo_type"] == "model"


@pytest.mark.parametrize("uri", ["hf://nope", "hf://datasets/aidigestorg/ai-village/", "hf://"])
def test_bad_uri_is_rejected(uri):
    with pytest.raises(ValueError):
        remote.resolve(uri)


def test_ingest_from_hf_uri_records_the_uri_as_source(monkeypatch, tmp_path):
    f = tmp_path / "chat_messages.jsonl.gz"
    with gzip.open(f, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"timestamp": 1781600000, "sender": "Claude", "content": "Found the donation link"}) + "\n")
        fh.write(json.dumps({"timestamp": 1781600060, "sender": "o3", "content": "thanks"}) + "\n")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kw: str(f))
    run = RunDir(tmp_path / "run")
    profile = pipeline.ingest("hf://datasets/aidigestorg/ai-village/chat_messages.jsonl.gz", run, adapter="auto")
    assert profile["source"] == "hf://datasets/aidigestorg/ai-village/chat_messages.jsonl.gz"
    assert profile["adapter"] == "chat" and profile["n_events"] == 2
    ev = run.read("events")
    assert set(ev["site"]) == {"chat_messages"}


def test_site_strips_compound_suffixes(tmp_path):
    f = tmp_path / "village-transcript.jsonl.gz"
    with gzip.open(f, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": "2026-06-01T00:00:00Z", "author": "A", "text": "hi"}) + "\n")
    assert ChatAdapter().load(f).events["site"].iloc[0] == "village-transcript"
