"""``swarmprov export``: a run's tables as CSV or Parquet files, for R, a spreadsheet or another pipeline.

Mapping-specific tables are written as ``<table>_<mapping>.<ext>``.  ``exposures`` and ``edges_exposure_citation``
keep the columns of the CSV mirrors committed under ``results/`` unless ``all_columns`` is set.
"""

from __future__ import annotations

from pathlib import Path

from .schema import REPORT_TABLES, RunDir

EXPOSURE_COLUMNS = ["claim_id", "agent", "family", "episode", "item", "value_norm", "t_report", "t_public", "src_agent",
                    "t_self_prepared", "gap_s", "D", "D_w10m", "D_w60m", "self_prepared", "D_cons", "D_cons_w60m",
                    "y_instant", "latency_known", "y_consensus", "consensus_value", "latency_class", "wrong_flag",
                    "t_public_visible", "D_visible", "src_deleted_before_report", "t_src_deleted", "item_fe"]
EDGE_COLUMNS = ["src", "dst", "family", "item", "value_norm", "kind", "dt_s", "event_id", "within_clock_res"]

# name -> (the stage table it reads, per mapping, the mirror's columns or None for all, edge kinds kept or None)
EXPORTS = {
    "exposures": ("exposures", True, EXPOSURE_COLUMNS, None),
    "edges": ("edges", True, None, None),
    "edges_exposure_citation": ("edges", True, EDGE_COLUMNS, ("exposure", "citation")),
    "chains": ("chains", True, None, None),
    "agent_profiles": ("agent_profiles", True, None, None),
    **{t: (t, True, None, None) for t in REPORT_TABLES},
    **{t: (t, False, None, None) for t in ("events", "agents", "claims", "tags", "mentions")},
}
DEFAULT = ("exposures", "edges_exposure_citation", "adoptions", "disputes", "a1_by_family")
STAGES = {"events": "ingest", "agents": "extract", "claims": "extract", "tags": "extract", "mentions": "expose",
          "exposures": "expose", "edges": "graph", "chains": "graph", "agent_profiles": "graph"}   # else report
NONE_WHEN = {"adoptions": "no configured technique appears in a timed post", "a1_by_family": "there are no agent-rounds",
             "disputes": "no item-value mention has a timestamp"}


def export(run_path, out, tables=DEFAULT, fmt: str = "csv", mapping: str = "merged", all_columns: bool = False) -> int:
    """Write each of ``tables`` (names of ``EXPORTS``) to ``out``; a table the run lacks is named and skipped.
    Returns 0, or 1 when none could be written."""
    if not (Path(run_path) / "profile.json").exists():
        raise SystemExit(f"{run_path}: not a run directory (no profile.json); `swarmprov run` makes one")
    run, out = RunDir(run_path), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for name in tables:
        table, per_mapping, cols, kinds = EXPORTS[name]
        src = f"{table}_{mapping}" if per_mapping else table
        if not run.has(src):
            print(f"[swarmprov] {name}: no {src}.parquet in {run.path} (written by `swarmprov {STAGES.get(table, 'report')}`"
                  + (f", and not when {NONE_WHEN[table]}" if table in NONE_WHEN else "") + "), skipped")
            continue
        df = run.read(src)
        if kinds:
            df = df[df["kind"].isin(kinds)]
        if cols and not all_columns:
            df = df[[c for c in cols if c in df.columns]]
        path = out / f"{name}_{mapping}.{fmt}" if per_mapping else out / f"{name}.{fmt}"
        df.to_csv(path, index=False) if fmt == "csv" else df.to_parquet(path, index=False)
        print(f"[swarmprov] wrote {path} ({len(df):,} rows)")
        written += 1
    return 0 if written else 1
