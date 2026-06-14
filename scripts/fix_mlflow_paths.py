"""Rewrite absolute artifact paths in an MLflow SQLite DB to the LOCAL mlruns/.

Why: the iBOT pretrain (and its encoder_50000/100000 snapshots) was logged on another
machine, so the DB stores absolute paths like /home/infres/adurand-25/.../mlruns/... .
On any other host mlflow can't resolve `runs:/<run>/encoder_50000` → "model not found",
even though the files ARE present under ./mlruns/1/models/m-*.

This rewrites every absolute ".../mlruns" prefix to `$(pwd)/mlruns` across all path columns.

Usage (run from the project root, ON the machine that will run the HPO):
    python scripts/fix_mlflow_paths.py mlflow.db            # dry-run (prints changes)
    python scripts/fix_mlflow_paths.py mlflow.db --apply    # actually write (backs up first)
"""
import os
import shutil
import sqlite3
import sys

DB = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "mlflow.db"
APPLY = "--apply" in sys.argv
NEW_MLRUNS = os.path.abspath("mlruns")

TARGETS = [
    ("logged_models", "artifact_location"),
    ("model_versions", "storage_location"),
    ("model_versions", "source"),
    ("runs", "artifact_uri"),
    ("experiments", "artifact_location"),
    ("tags", "value", "key = 'mlflow.artifactUri'"),
]
_SCHEMES = ("file://", "mlflow-artifacts:")


def rewrite(val: str) -> str:
    if not isinstance(val, str):
        return val
    scheme, body = "", val
    for s in _SCHEMES:
        if body.startswith(s):
            scheme, body = s, body[len(s):]
            break
    i = body.rfind("/mlruns")
    if i < 0:
        return val
    return scheme + NEW_MLRUNS + body[i + len("/mlruns"):]


def main() -> None:
    print(f"DB={DB}  →  new mlruns root = {NEW_MLRUNS}  ({'APPLY' if APPLY else 'DRY-RUN'})")
    if not os.path.isdir(NEW_MLRUNS):
        print(f"  WARNING: {NEW_MLRUNS} does not exist — are you in the project root?")
    if APPLY:
        bak = DB + ".bak"
        shutil.copy(DB, bak)
        print(f"  backup → {bak}")
    con = sqlite3.connect(DB)
    cur = con.cursor()
    total = 0
    for tbl, col, *where in TARGETS:
        wsql = f" AND ({where[0]})" if where else ""
        try:
            rows = cur.execute(
                f"SELECT rowid, {col} FROM {tbl} WHERE {col} LIKE '%/mlruns%'{wsql}"
            ).fetchall()
        except sqlite3.OperationalError as e:
            print(f"  skip {tbl}.{col}: {e}")
            continue
        changed = 0
        for rid, val in rows:
            nv = rewrite(val)
            if nv != val:
                changed += 1
                if changed <= 1:
                    print(f"  {tbl}.{col}: e.g.\n     {val}\n  -> {nv}")
                if APPLY:
                    cur.execute(f"UPDATE {tbl} SET {col} = ? WHERE rowid = ?", (nv, rid))
        if changed:
            print(f"  {tbl}.{col}: {changed} rows {'updated' if APPLY else 'would change'}")
        total += changed
    if APPLY:
        con.commit()
    con.close()
    print(f"Total: {total} rows {'updated' if APPLY else 'to update'}."
          f"{'' if APPLY else '  Re-run with --apply to commit.'}")


if __name__ == "__main__":
    main()
