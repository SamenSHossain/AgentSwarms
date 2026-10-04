"""The Swarm traces redacted payload release: classified as a reconstruction
corpus (no wall clock, no agent identity), reported as a tree + redaction
coverage, with the provenance analyses correctly absent."""

import gzip
import json
from pathlib import Path

import pytest

from swarmprov import adapters, pipeline
from swarmprov.adapters.swarmtraces import SwarmTracesAdapter, _category
from swarmprov.schema import RunDir


def _rows():
    # two payloads; one has a response + a recovered_text child, the other a child whose
    # parent is absent (an orphan). One recovered_text repeats another's exact body.
    return [
        {"id": "R0000001", "cite": "R0000001:aa", "kind": "payload", "parent_id": None, "time_utc": None,
         "tags": "", "text": "fetch('[SERVICE HOST 1]/[SERVICE 2 URL 1]',{headers:{'X-API-Key':'[CREDENTIAL 1]'}})"},
        {"id": "R0000002", "cite": "R0000002:bb", "kind": "response", "parent_id": "R0000001", "time_utc": None,
         "tags": "article-evidence;appendix-request", "text": "200 OK [ENCODED BLOB 5]"},
        {"id": "R0000003", "cite": "R0000003:cc", "kind": "recovered_text", "parent_id": "R0000001", "time_utc": None,
         "tags": "", "text": "shared body"},
        {"id": "R0000004", "cite": "R0000004:dd", "kind": "recovered_text", "parent_id": "R0000999", "time_utc": None,
         "tags": "", "text": "shared body"},
        {"id": "R0000005", "cite": "R0000005:ee", "kind": "payload", "parent_id": None, "time_utc": None,
         "tags": "", "text": "x+='[SHORTENER CODE 225495]'; [REDACTED:destination:000002]"},
    ]


@pytest.fixture
def release(tmp_path: Path) -> Path:
    f = tmp_path / "redacted.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in _rows()) + "\n")
    return f


def test_category_strips_the_instance_number():
    assert _category("[CREDENTIAL 1]") == "CREDENTIAL"
    assert _category("[ENCODED BLOB 207876]") == "ENCODED BLOB"
    assert _category("[SERVICE 2 URL 1]") == "SERVICE URL"
    assert _category("[REDACTED:destination:000002]") == "destination"
    assert _category("[REDACTED:runtime_identifier]") == "runtime_identifier"
    assert _category("[000123]") == "(unlabelled)"


def test_detected_as_a_file_a_dir_and_a_gz(release: Path, tmp_path: Path):
    assert adapters.detect(release).name == "swarmtraces"
    assert adapters.detect(release.parent).name == "swarmtraces"          # directory holding redacted.jsonl
    gz = tmp_path / "gzdir" / "redacted.jsonl.gz"
    gz.parent.mkdir()
    gz.write_bytes(gzip.compress(release.read_bytes()))
    assert adapters.detect(gz).name == "swarmtraces"
    assert adapters.detect(gz.parent).name == "swarmtraces"


def test_a_message_log_is_not_mistaken_for_a_release(tmp_path: Path):
    # a chat transcript (no id/cite/kind/parent_id/time_utc/tags columns) must not sniff as swarmtraces
    chat = tmp_path / "chat.jsonl"
    chat.write_text(json.dumps({"ts": "2026-06-01T00:00:00Z", "author": "A", "text": "hi"}) + "\n")
    assert not SwarmTracesAdapter().sniff(chat)


def test_release_classified_with_no_clock_and_no_author(release: Path):
    b = adapters.get("swarmtraces").load(release)
    assert len(b.events) == 0                                   # no timed posts: the pipeline writes a context report
    assert b.capabilities.has_wall_clock is False
    assert b.capabilities.has_explicit_author is False
    st = b.notes["swarmtraces"]
    assert st["n_records"] == 5
    assert st["by_kind"] == {"payload": 2, "recovered_text": 2, "response": 1}
    assert st["integrity"] == {"ids_unique": True, "cites_unique": True, "cite_prefixes_match_id": True}


def test_reconstruction_tree_counts_parents_and_orphans(release: Path):
    st = adapters.get("swarmtraces").load(release).notes["swarmtraces"]
    tree = st["tree"]
    assert tree["roots"] == 2 and tree["children"] == 3
    assert tree["child_of_parent_kind"] == {"response <- payload": 1, "recovered_text <- payload": 1}
    assert tree["parents_with_children"] == 1          # only R0000001 has children present in the release
    assert tree["orphan_children"] == 1                # R0000004 points at an absent R0000999
    assert tree["max_depth"] == 1


def test_tags_text_and_redaction(release: Path):
    st = adapters.get("swarmtraces").load(release).notes["swarmtraces"]
    assert st["tags"]["n_tagged"] == 1 and st["tags"]["families"] == {"article-evidence": 1}
    assert st["text"]["distinct"] == 4 and st["text"]["repeated"] == 1    # "shared body" appears twice
    red = st["redaction"]
    assert red["records_with_placeholder"] == 3    # R1 (3), R2 (1), R5 (2)
    assert red["placeholder_occurrences"] == 6
    assert set(red["top_categories"]) == {"SERVICE HOST", "SERVICE URL", "CREDENTIAL", "ENCODED BLOB",
                                           "SHORTENER CODE", "destination"}


def test_full_run_is_a_reconstruction_report_without_provenance(release: Path, tmp_path: Path):
    run = pipeline.run_all(release, str(tmp_path / "run"))
    md = (run.path / "report.md").read_text()
    assert "Reconstruction report" in md
    assert "189" not in md  # the synthetic release, not the real 189,579-record one
    assert "5 recovered records" in md
    assert "no wall clock and no agent identity" in md
    # the provenance sections must be absent: there is nothing to order or attribute
    for heading in ("A1. Provenance", "A3. Technique diffusion", "A5. Structure"):
        assert heading not in md
    # reconstruction facts are present
    assert "Reconstruction tree" in md and "reconstruction parent" in md
    assert "CREDENTIAL" in md
    # summary.json carries the structured facts, not a truncated repr
    summary = json.loads((run.path / "summary.json").read_text())
    assert summary["swarmtraces"]["by_kind"]["payload"] == 2
    assert summary["swarmtraces"]["tree"]["orphan_children"] == 1


def test_zip_release_is_read(release: Path, tmp_path: Path):
    import zipfile
    z = tmp_path / "bundle.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(release, "evidence/redacted.jsonl")
    assert adapters.detect(z).name == "swarmtraces"
    st = adapters.get("swarmtraces").load(z).notes["swarmtraces"]
    assert st["n_records"] == 5
