"""Check the released original best.pt files against the server hashes."""
from pathlib import Path
import hashlib
import json
import sys

root = Path(__file__).resolve().parents[1]
records = json.loads((root / "artifacts/experiments.json").read_text(encoding="utf-8"))
failed = 0
for record in records:
    path = root / record["checkpoint"]
    if not path.is_file():
        print("MISSING", record["checkpoint"]); failed += 1; continue
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    ok = h.hexdigest() == record["checkpoint_sha256"]
    print("OK" if ok else "MISMATCH", record["run_id"])
    failed += not ok
sys.exit(1 if failed else 0)
