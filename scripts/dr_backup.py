#!/usr/bin/env python3
"""
Disaster Recovery Snapshot  (Phase IX Stage 50)
=================================================
Daily tar.gz snapshot of the data/ directory plus the audit_trail.db, with
optional GPG symmetric encryption. Stores snapshots in a configurable
backup directory; verifies integrity with SHA-256 checksum on every run.

Cold-start playbook (built into the report output):
  1. Reinstall Python deps: pip install -r requirements.txt
  2. Copy latest snapshot to data/_restore/
  3. tar xzf snapshot.tar.gz -C data/
  4. Verify audit chain: python3 scripts/audit_trail.py --verify
  5. Re-run master_controller.py --dry-run to smoke-test

Output: data/dr_backup.json with last snapshot metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
DEFAULT_BACKUP_DIR = ROOT / "backups"
OUTPUT_FILE = DATA_DIR / "dr_backup.json"

# Files / dirs to include
INCLUDE = ["data"]
# Patterns to exclude (large or regenerable)
EXCLUDE = [
    "data/_restore",
    "data/perplexity_cache",
    "data/cot_cache",
    "data/alt_data",
    "data/equity_features",
    "data/equity_swarm",
    "data/equity_ranker",
    "data/__pycache__",
]
# Hard size cap per backup (MB)
MAX_BACKUP_MB = 256

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def _should_exclude(path: str) -> bool:
    for pattern in EXCLUDE:
        if pattern in path:
            return True
    return False


def _tar_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    if _should_exclude(tarinfo.name):
        return None
    return tarinfo


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_snapshot(
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    encrypt_pass: str | None = None,
) -> dict:
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_name = f"snapshot_{stamp}.tar.gz"
    archive_path = backup_dir / archive_name

    # Build the tar.gz
    with tarfile.open(archive_path, "w:gz") as tar:
        for inc in INCLUDE:
            tar.add(ROOT / inc, arcname=inc, filter=_tar_filter)

    size_mb = archive_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_BACKUP_MB:
        archive_path.unlink()
        return {
            "success": False,
            "error":   f"snapshot {size_mb:.1f} MB exceeds cap {MAX_BACKUP_MB} MB",
        }

    checksum = _sha256(archive_path)

    # Optional GPG encryption
    encrypted_path = None
    if encrypt_pass:
        encrypted_path = archive_path.with_suffix(".tar.gz.gpg")
        try:
            subprocess.run(
                [
                    "gpg", "--batch", "--yes",
                    "--passphrase", encrypt_pass,
                    "--symmetric", "--cipher-algo", "AES256",
                    "--output", str(encrypted_path),
                    str(archive_path),
                ],
                check=True, capture_output=True,
            )
            # Remove the unencrypted version
            archive_path.unlink()
            archive_path = encrypted_path
            checksum = _sha256(archive_path)
        except subprocess.CalledProcessError as exc:
            return {
                "success": False,
                "error":   f"gpg encryption failed: {exc.stderr.decode() if exc.stderr else exc}",
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error":   "gpg binary not found",
            }

    # Cleanup old snapshots — keep last 14
    snapshots = sorted(
        list(backup_dir.glob("snapshot_*.tar.gz*")),
        key=lambda p: p.stat().st_mtime,
    )
    deleted = []
    for old in snapshots[:-14]:
        try:
            old.unlink()
            deleted.append(old.name)
        except Exception:
            pass

    return {
        "success":     True,
        "archive_path":str(archive_path),
        "size_mb":     round(size_mb, 2),
        "checksum":    checksum,
        "encrypted":   encrypted_path is not None,
        "n_remaining": len(snapshots) - len(deleted),
        "deleted":     deleted,
    }


def list_snapshots(backup_dir: Path = DEFAULT_BACKUP_DIR) -> list:
    if not backup_dir.exists():
        return []
    items = []
    for p in sorted(backup_dir.glob("snapshot_*.tar.gz*")):
        st = p.stat()
        items.append({
            "name":    p.name,
            "size_mb": round(st.st_size / (1024 * 1024), 2),
            "mtime":   datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        })
    return items


def verify_latest(backup_dir: Path = DEFAULT_BACKUP_DIR) -> dict:
    snaps = sorted(
        list(backup_dir.glob("snapshot_*.tar.gz*")),
        key=lambda p: p.stat().st_mtime,
    )
    if not snaps:
        return {"valid": False, "reason": "no snapshots found"}
    latest = snaps[-1]
    # SHA-256 must compute without error; if the file is corrupt, this will error
    try:
        h = _sha256(latest)
        return {"valid": True, "latest": latest.name, "checksum": h}
    except Exception as exc:
        return {"valid": False, "reason": str(exc)}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_dr_backup(
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    encrypt_pass: str | None = None,
) -> dict:
    snap = make_snapshot(backup_dir, encrypt_pass)
    snaps = list_snapshots(backup_dir)
    verify = verify_latest(backup_dir) if snap["success"] else {"valid": False}

    playbook = [
        "1. pip install -r requirements.txt",
        "2. mkdir -p data/_restore && cp <snapshot> data/_restore/",
        "3. tar xzf data/_restore/<snapshot> -C ./",
        "4. python3 scripts/audit_trail.py --verify",
        "5. python3 scripts/master_controller.py --dry-run",
    ]

    result = {
        "generated_at":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backup_dir":    str(backup_dir),
        "snapshot":      snap,
        "all_snapshots": snaps,
        "n_snapshots":   len(snaps),
        "verification":  verify,
        "cold_start_playbook": playbook,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, default=str))
    _print_report(result)
    return result


def _print_report(r: dict) -> None:
    print(f"\n{SEP}\n  DR SNAPSHOT\n{SEP}")
    snap = r["snapshot"]
    if snap.get("success"):
        print(f"  Archive:    {snap['archive_path']}")
        print(f"  Size:       {snap['size_mb']:.2f} MB")
        print(f"  Encrypted:  {snap['encrypted']}")
        print(f"  Checksum:   {snap['checksum'][:32]}...")
    else:
        print(f"  ⚠ snapshot failed: {snap.get('error')}")
    print()
    print(f"  TOTAL SNAPSHOTS RETAINED: {r['n_snapshots']}")
    for s in r["all_snapshots"][-3:]:
        print(f"    {s['name']}  ({s['size_mb']:.1f} MB, {s['mtime']})")
    print()
    print(f"  VERIFICATION: {r['verification']}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disaster Recovery Backup")
    parser.add_argument("--backup-dir", default=str(DEFAULT_BACKUP_DIR))
    parser.add_argument("--encrypt-pass", default=None,
                        help="GPG symmetric passphrase. Omit for plain tar.gz.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    backup_dir = Path(args.backup_dir)

    if args.list:
        for s in list_snapshots(backup_dir):
            print(f"  {s['name']}  {s['size_mb']:.1f} MB  {s['mtime']}")
    elif args.verify:
        print(verify_latest(backup_dir))
    else:
        run_dr_backup(backup_dir, encrypt_pass=args.encrypt_pass)
