"""Resolve remote dataset URIs to local paths before an adapter sees them.

``hf://datasets/<owner>/<name>[@<revision>]/<path>`` is the form pandas and
fsspec use.  A file URI downloads that one file; a repo URI (no file part)
downloads the whole dataset so a multi-file adapter can sniff the directory.
Files land in the huggingface_hub cache (``HF_HOME``), so repeat runs are free.
"""

from __future__ import annotations

import re
from pathlib import Path

HF_RX = re.compile(
    r"^hf://(?:(?P<type>datasets|spaces)/)?(?P<repo>(?!datasets/|spaces/)[^/@]+/[^/@]+)"
    r"(?:@(?P<rev>[^/]+))?(?:/(?P<file>.+[^/]))?$"
)
REPO_TYPES = {"datasets": "dataset", "spaces": "space", None: "model"}


def is_remote(path) -> bool:
    return isinstance(path, str) and path.startswith("hf://")


def resolve(path: str | Path, cache_dir: str | None = None) -> Path:
    """Local path for ``path``: downloads an hf:// URI, returns anything else unchanged."""
    if not is_remote(path):
        return Path(path)
    m = HF_RX.match(path)
    if not m:
        raise ValueError(f"not a Hugging Face URI: {path!r} (expected hf://datasets/<owner>/<name>/<file>)")
    try:
        import huggingface_hub as hub
    except ImportError as e:  # pragma: no cover
        raise ImportError("hf:// inputs need huggingface_hub: pip install 'swarmprov[hf]'") from e
    kw = dict(repo_id=m["repo"], repo_type=REPO_TYPES[m["type"]], revision=m["rev"], cache_dir=cache_dir)
    if m["file"]:
        return Path(hub.hf_hub_download(filename=m["file"], **kw))
    return Path(hub.snapshot_download(**kw))
