import gzip
import json
import sys
import types

import pytest

from swarmprov import pipeline, remote
from swarmprov.adapters.chat import ChatAdapter
from swarmprov.schema import RunDir

# remote.resolve imports huggingface_hub lazily, so a stub module keeps these tests independent of the extra
huggingface_hub = types.SimpleNamespace(hf_hub_download=None, snapshot_download=None)


@pytest.fixture(autouse=True)
def _stub_hub(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)


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


@pytest.mark.parametrize("uri,rev,filename", [
    ("hf://datasets/o/n@refs/convert/parquet/default/train/0000.parquet", "refs/convert/parquet", "default/train/0000.parquet"),
    ("hf://datasets/o/n@refs/pr/3/file.txt", "refs/pr/3", "file.txt"),
    ("hf://datasets/o/n@refs%2Fconvert%2Fparquet/f.parquet", "refs/convert/parquet", "f.parquet"),
    ("hf://datasets/o/n@v1.0/x", "v1.0", "x"),
])
def test_revisions_with_slashes_and_encoding(monkeypatch, uri, rev, filename):
    calls = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kw: calls.append(kw) or "/x")
    remote.resolve(uri)
    assert (calls[0]["revision"], calls[0]["filename"]) == (rev, filename)


def test_models_prefix_and_repo_revision(monkeypatch):
    calls = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kw: calls.append(kw) or "/x")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: calls.append(kw) or "/d")
    remote.resolve("hf://models/o/n/config.json")
    assert (calls[0]["repo_id"], calls[0]["repo_type"], calls[0]["filename"]) == ("o/n", "model", "config.json")
    remote.resolve("hf://datasets/o/n@refs/pr/3")
    assert calls[1] == {"repo_id": "o/n", "repo_type": "dataset", "revision": "refs/pr/3", "cache_dir": None}


@pytest.mark.parametrize("uri", ["hf://nope", "hf://datasets/aidigestorg/ai-village/", "hf://", "hf://datasets/o"])
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


MSGS = [{"timestamp": 1781600000, "sender": "Claude", "content": "Found the donation link"},
        {"timestamp": 1781600060, "sender": "o3", "content": "thanks"}]


@pytest.mark.parametrize("name,body", [
    ("one.jsonl", json.dumps(MSGS[0]) + "\n"),
    ("lines.jsonl", "".join(json.dumps(m) + "\r\n" for m in MSGS) + "\n"),
    ("pretty.jsonl", json.dumps(MSGS, indent=2)),
    ("oneline-array.jsonl", json.dumps(MSGS) + "\n"),
    ("wrapped.jsonl", json.dumps({"messages": MSGS})),
    ("doc.json", json.dumps({"messages": MSGS}, indent=2)),
])
def test_chat_adapter_reads_every_json_shape(tmp_path, name, body):
    f = tmp_path / name
    f.write_text(body)
    a = ChatAdapter()
    assert a.sniff(f)
    assert len(a.load(f).events) == (1 if name == "one.jsonl" else 2)


def test_chat_adapter_rejects_textless_input_and_directories(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text(json.dumps({"ts": 1, "author": "A", "foo": "bar"}) + "\n")
    with pytest.raises(ValueError, match="no messages with text"):
        ChatAdapter().load(f)
    with pytest.raises(ValueError, match="directory"):
        ChatAdapter().load(tmp_path)
    assert not ChatAdapter().sniff(tmp_path)
