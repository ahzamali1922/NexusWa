"""
NexusWatch — Ingestion Pipeline  (Hardware-Optimized Build)
============================================================
Target Machine:
    CPU  : Intel Xeon W-2245  (8C / 16T @ 3.9 GHz)
    RAM  : 64 GB  → full CSV loaded in one shot, no chunking
    SSD  : 2 TB NVMe  → Parquet I/O is near-instant
    GPU  : RTX A6000 48 GB VRAM  → full graph on GPU later

Tasks:
    1. accounts CSV  → accounts.parquet            (clean, typed)
    2. trans CSV     → transactions_clean.parquet  (orphan-dropped, typed)
    3. patterns .txt → topology_mapping.parquet    (streaming parser)
    4. Validation    → zero-orphan assertion + shape report

Confirmed Topologies (audited across all 5 pattern files):
    1 STACK           5 SCATTER-GATHER
    2 CYCLE           6 GATHER-SCATTER
    3 FAN-IN          7 BIPARTITE
    4 FAN-OUT         8 RANDOM

Run:
    python nexuswatch_ingestion.py

Install:
    pip install pandas pyarrow tqdm
"""

import os
import re
import time
import multiprocessing
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ─────────────────────────────────────────────────────────────────────────────
# HARDWARE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CPU_CORES = multiprocessing.cpu_count()   # 16 on Xeon W-2245

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  —  point DATA_DIR at your CSV folder
# ─────────────────────────────────────────────────────────────────────────────
PREFIX = os.environ.get("DATASET_PREFIX", "HI-Medium")

DATA_DIR = Path(__file__).resolve().parent / "Dataset"
print(f"📂 Using dataset path: {DATA_DIR}")
OUT_DIR    = Path(f"./output/{PREFIX}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACCT_FILE  = DATA_DIR / f"{PREFIX}_accounts.csv"
TRANS_FILE = DATA_DIR / f"{PREFIX}_Trans.csv"
PATT_FILE  = DATA_DIR / f"{PREFIX}_Patterns.txt"

# ─────────────────────────────────────────────────────────────────────────────
# TOPOLOGY MAP
# Ground-truth: audited by grepping all 5 pattern files.
# Sub-variants ("CYCLE: Max 5 hops") collapsed to base key via regex.
# STAR does not exist in any file — confirmed.
# ─────────────────────────────────────────────────────────────────────────────
TOPOLOGY_MAP: dict[str, int] = {
    "STACK"          : 1,   # Layered chain A->B->C->D
    "CYCLE"          : 2,   # Circular flow back to origin, 2-13 hops
    "FAN-IN"         : 3,   # Many sources -> one account, 1-16 degree
    "FAN-OUT"        : 4,   # One source -> many accounts, 1-16 degree
    "SCATTER-GATHER" : 5,   # Disperse then recollect
    "GATHER-SCATTER" : 6,   # Collect then disperse, 1-16 degree
    "BIPARTITE"      : 7,   # Two-group structured layering
    "RANDOM"         : 8,   # Random walk chain, 1-12 hops
}

SEP  = "=" * 70
SEP2 = "-" * 70


def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def subsection(title: str):
    print(f"\n  -- {title}")


def elapsed(t0: float) -> str:
    return f"{time.time() - t0:.2f}s"


# =============================================================================
# TASK 1 — ACCOUNTS
# =============================================================================

def ingest_accounts() -> frozenset:
    """
    Loads full accounts CSV into RAM (141 MB — trivial on 64 GB).
    Cleans IDs, extracts Entity Type, saves accounts.parquet.
    Returns a frozenset of valid Account Numbers for orphan-drop downstream.
    """
    section("TASK 1 — Accounts Ingestion")
    t0 = time.time()

    # Full load with pyarrow engine (multi-threaded CSV parsing)
    df = pd.read_csv(ACCT_FILE, dtype=str, engine="pyarrow")
    df.columns = df.columns.str.strip()
    raw_count  = len(df)
    print(f"\n  Raw rows loaded    : {raw_count:>10,}  ({elapsed(t0)})")
    print(f"  Columns            : {list(df.columns)}")

    # Clean Account Number
    df["Account Number"] = df["Account Number"].str.strip().str.upper()

    # Drop nulls
    before = len(df)
    df.dropna(subset=["Account Number"], inplace=True)
    n_null = before - len(df)
    if n_null:
        print(f"  WARNING: Null Account Numbers dropped  : {n_null:,}")

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(subset=["Account Number"], keep="first", inplace=True)
    n_dupe = before - len(df)
    if n_dupe:
        print(f"  WARNING: Duplicate Account Numbers dropped : {n_dupe:,}")

    # Feature: extract Entity Type from Entity Name
    # e.g. "Corporation #183669" -> "Corporation"
    df["Entity Type"] = (
        df["Entity Name"]
        .str.extract(r"^([A-Za-z ]+)", expand=False)
        .str.strip()
    )

    # Feature: Bank ID as integer
    df["Bank ID"] = pd.to_numeric(df["Bank ID"], errors="coerce")

    # Save
    out_path = OUT_DIR / "accounts.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")

    size_mb = out_path.stat().st_size / 1_048_576
    print(f"\n  Clean rows saved   : {len(df):>10,}")
    print(f"  Unique banks       : {df['Bank Name'].nunique():>10,}")
    print(f"  Entity type dist   :")
    for etype, cnt in df["Entity Type"].value_counts().items():
        print(f"      {etype:<35} : {cnt:>8,}")
    print(f"\n  Parquet size       : {size_mb:.1f} MB")
    print(f"  Saved -> {out_path}  ({elapsed(t0)})")

    return frozenset(df["Account Number"].unique())


# =============================================================================
# TASK 2 — TRANSACTIONS  (full load, no chunking — 64 GB RAM)
# =============================================================================

def _fix_duplicate_account_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    The raw CSV has two columns both named 'Account'.
    Renames them positionally to 'From Account' and 'To Account'.
    Must be called immediately after read_csv before any other operation.
    """
    cols     = list(df.columns)
    acct_idx = [i for i, c in enumerate(cols) if c == "Account"]

    if len(acct_idx) != 2:
        raise ValueError(
            f"Expected exactly 2 'Account' columns, found {len(acct_idx)}.\n"
            f"Full column list: {cols}\n"
            f"Verify TRANS_FILE points to the correct CSV."
        )

    cols[acct_idx[0]] = "From Account"
    cols[acct_idx[1]] = "To Account"
    df.columns = cols
    return df


def ingest_transactions(valid_ids: frozenset) -> int:
    """
    Loads full HI-Medium_Trans.csv (~2.96 GB) into RAM in one shot.
    On 64 GB this is safe and ~3-4x faster than chunked reading.

    Steps:
      1. Read with pyarrow engine  (16 Xeon threads, multi-threaded)
      2. Fix duplicate 'Account' column names
      3. Strip + uppercase Account IDs
      4. Drop null Account IDs
      5. DROP ORPHANS — vectorized isin() on frozenset
      6. Cast types: numeric amounts, datetime, Int8 label
      7. Derive time features (Hour, DayOfWeek, Month) for edge features later
      8. Save to Parquet, Snappy compressed
    """
    section("TASK 2 — Transactions Ingestion  (~2.96 GB, full load)")
    t0 = time.time()

    # Step 1: Full load
    print(f"\n  Loading full CSV into RAM (pyarrow engine, {CPU_CORES} threads)...")
    df = pd.read_csv(TRANS_FILE, dtype=str, engine="pyarrow")
    df.columns = df.columns.str.strip()
    raw_count  = len(df)
    ram_used   = df.memory_usage(deep=True).sum() / 1_073_741_824
    print(f"  Raw rows loaded    : {raw_count:>12,}  ({elapsed(t0)})")
    print(f"  RAM used by df     : {ram_used:.2f} GB  (of your 64 GB)")

    # Step 2: Fix duplicate column names
    df = _fix_duplicate_account_columns(df)
    print(f"  Columns fixed      : {list(df.columns)}")

    # Step 3: Strip + uppercase
    df["From Account"] = df["From Account"].str.strip().str.upper()
    df["To Account"]   = df["To Account"].str.strip().str.upper()

    # Step 4: Drop null Account IDs
    before = len(df)
    df.dropna(subset=["From Account", "To Account"], inplace=True)
    n_null = before - len(df)
    if n_null:
        print(f"  WARNING: Null Account IDs dropped  : {n_null:>12,}")

    # Step 5: ORPHAN DROP
    # Vectorized isin() on frozenset uses hash lookup — O(1) per element
    print(f"\n  Running orphan drop (vectorized isin on frozenset)...")
    t_orphan = time.time()
    mask     = (
        df["From Account"].isin(valid_ids) &
        df["To Account"].isin(valid_ids)
    )
    n_orphan = int((~mask).sum())
    df       = df[mask].copy()   # .copy() frees the boolean mask from memory
    print(f"  Orphaned rows dropped      : {n_orphan:>12,}  "
          f"({n_orphan / raw_count * 100:.4f}% of raw)  ({elapsed(t_orphan)})")
    print(f"  Clean rows remaining       : {len(df):>12,}")

    # Step 6: Type casting
    print(f"\n  Casting types...")
    t_cast = time.time()
    df["Amount Paid"]     = pd.to_numeric(df["Amount Paid"],     errors="coerce")
    df["Amount Received"] = pd.to_numeric(df["Amount Received"], errors="coerce")
    df["Is Laundering"]   = pd.to_numeric(df["Is Laundering"],   errors="coerce").astype("Int8")
    df["Timestamp"]       = pd.to_datetime(df["Timestamp"],      errors="coerce")
    print(f"  Type casting done  ({elapsed(t_cast)})")

    # Step 7: Derive time features (free edge features for the GNN later)
    df["Hour"]      = df["Timestamp"].dt.hour.astype("Int8")
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek.astype("Int8")
    df["Month"]     = df["Timestamp"].dt.month.astype("Int8")

    # Class imbalance report
    subsection("Class Imbalance (Is Laundering)")
    counts = df["Is Laundering"].value_counts().sort_index()
    total  = len(df)
    for label, cnt in counts.items():
        tag = "LAUNDERING" if label == 1 else "LEGITIMATE"
        print(f"      {tag:<12} : {cnt:>12,}  ({cnt / total * 100:.4f}%)")
    ratio = counts.get(0, 1) / max(counts.get(1, 1), 1)
    print(f"      Imbalance ratio (legit:fraud)  =  {ratio:.1f} : 1")
    print(f"      -> Set pos_weight={ratio:.0f} in BCEWithLogitsLoss")

    # Payment format distribution
    subsection("Payment Format Distribution")
    for fmt, cnt in df["Payment Format"].value_counts().items():
        print(f"      {fmt:<20} : {cnt:>10,}  ({cnt / total * 100:.2f}%)")

    # Step 8: Save
    print(f"\n  Saving to Parquet (Snappy)...")
    t_save   = time.time()
    out_path = OUT_DIR / "transactions_clean.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    size_mb  = out_path.stat().st_size / 1_048_576
    print(f"  Parquet size       : {size_mb:.1f} MB  (was ~2,960 MB raw)")
    print(f"  Compression ratio  : {2960 / size_mb:.1f}x")
    print(f"  Saved -> {out_path}  ({elapsed(t_save)})")
    print(f"\n  Total task time    : {elapsed(t0)}")

    return len(df)


# =============================================================================
# TASK 3 — TOPOLOGY PARSER  (.txt -> Global Mapping)
# =============================================================================

def parse_topology_patterns() -> pd.DataFrame:
    """
    Streams HI-Medium_Patterns.txt line by line (2.2 MB).
    Builds the global answer key:  (From_Bank, To_Bank) -> Topology_ID

    Format of each block:
        BEGIN LAUNDERING ATTEMPT - <TOPOLOGY>[: <sub-variant info>]
        <Timestamp>,<From_Bank>,<From_Acct>,<To_Bank>,<To_Acct>,...
        END LAUNDERING ATTEMPT - <TOPOLOGY>

    Regex captures BASE name only: "CYCLE: Max 5 hops" -> "CYCLE"
    """
    section("TASK 3 — Streaming Topology Parser")
    t0 = time.time()

    # [A-Z][A-Z\-]* stops at ':' or whitespace — captures base name only
    begin_re = re.compile(
        r"BEGIN\s+LAUNDERING\s+ATTEMPT\s*[-]\s*([A-Z][A-Z\-]*)",
        re.IGNORECASE
    )

    records:          list[tuple] = []
    current_topology: str | None  = None
    blocks_found:     int         = 0
    tx_lines_parsed:  int         = 0
    unknown_topos:    set         = set()

    fsize_kb = PATT_FILE.stat().st_size / 1024
    print(f"\n  Streaming : {PATT_FILE}  ({fsize_kb:.1f} KB)")

    with open(PATT_FILE, "r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line  = raw_line.strip()
            if not line:
                continue
            upper = line.upper()

            if upper.startswith("BEGIN LAUNDERING"):
                m = begin_re.search(line)
                if m:
                    current_topology = m.group(1).strip().upper()
                    blocks_found    += 1
                continue

            if upper.startswith("END LAUNDERING"):
                current_topology = None
                continue

            if current_topology is not None:
                parts = line.split(",")
                if len(parts) < 5:
                    continue
                from_bank = parts[1].strip()
                to_bank   = parts[3].strip()
                topo_id   = TOPOLOGY_MAP.get(current_topology)
                if topo_id is None:
                    unknown_topos.add(current_topology)
                    topo_id = -1
                records.append((from_bank, to_bank, current_topology, topo_id))
                tx_lines_parsed += 1

    mapping_df = pd.DataFrame(
        records,
        columns=["From_Bank", "To_Bank", "Topology_Name", "Topology_ID"]
    )

    print(f"\n  Blocks found              : {blocks_found:>8,}")
    print(f"  Transaction lines parsed  : {tx_lines_parsed:>8,}")
    print(f"  Raw (Bank, Bank) pairs    : {len(records):>8,}")

    if unknown_topos:
        print(f"\n  WARNING — Unknown topologies (add to TOPOLOGY_MAP):")
        for t in sorted(unknown_topos):
            print(f"      \"{t}\"")

    subsection("Topology Distribution (tx lines per type)")
    dist    = mapping_df["Topology_Name"].value_counts()
    max_cnt = dist.max()
    for name, cnt in dist.items():
        tid = TOPOLOGY_MAP.get(name, -1)
        bar = "#" * int(cnt / max_cnt * 35)
        print(f"      [{tid}] {name:<20} {bar:<35} {cnt:>6,}")

    # Deduplicate: keep most-frequent topology per (From_Bank, To_Bank) pair
    unique_mapping = (
        mapping_df
        .groupby(["From_Bank", "To_Bank", "Topology_Name", "Topology_ID"])
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
        .drop_duplicates(subset=["From_Bank", "To_Bank"], keep="first")
        .drop(columns=["Count"])
        .reset_index(drop=True)
    )

    print(f"\n  Unique (From_Bank, To_Bank) pairs : {len(unique_mapping):>6,}")

    out_path = OUT_DIR / "topology_mapping.parquet"
    unique_mapping.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    print(f"  Saved -> {out_path}  ({elapsed(t0)})")

    return unique_mapping


# =============================================================================
# TASK 4 — VALIDATION
# =============================================================================

def validate_outputs() -> bool:
    section("TASK 4 — Validation & Sanity Checks")
    all_ok = True

    # File shape report
    for fname in ["accounts.parquet", "transactions_clean.parquet", "topology_mapping.parquet"]:
        path = OUT_DIR / fname
        if not path.exists():
            print(f"\n  MISSING : {fname}")
            all_ok = False
            continue
        df      = pd.read_parquet(path)
        size_mb = path.stat().st_size / 1_048_576
        print(f"\n  [{fname}]")
        print(f"    Rows     : {len(df):,}")
        print(f"    Columns  : {list(df.columns)}")
        print(f"    Size     : {size_mb:.1f} MB")

    # Zero-orphan assertion
    print(f"\n  {SEP2}")
    print(f"  ZERO-ORPHAN ASSERTION")
    print(f"  {SEP2}")

    acct_df  = pd.read_parquet(OUT_DIR / "accounts.parquet")
    trans_df = pd.read_parquet(OUT_DIR / "transactions_clean.parquet")
    valid    = frozenset(acct_df["Account Number"])

    from_orphans = set(trans_df["From Account"]) - valid
    to_orphans   = set(trans_df["To Account"])   - valid
    all_orphans  = from_orphans | to_orphans

    if len(all_orphans) == 0:
        print(f"\n  PASSED — Zero orphaned accounts in transaction file.")
        print(f"  Graph is 100% safe to build: every node has an account record.")
    else:
        print(f"\n  FAILED — {len(all_orphans):,} orphaned accounts remain!")
        print(f"     From-side orphans : {len(from_orphans):,}")
        print(f"     To-side orphans   : {len(to_orphans):,}")
        print(f"     Sample            : {list(all_orphans)[:5]}")
        all_ok = False

    # Label sanity
    labels      = trans_df["Is Laundering"].value_counts().sort_index()
    null_labels = trans_df["Is Laundering"].isna().sum()
    print(f"\n  Label distribution (post-clean):")
    for lbl, cnt in labels.items():
        print(f"      {lbl} -> {cnt:,}")
    if null_labels:
        print(f"  WARNING: {null_labels:,} null labels — investigate!")

    # Topology coverage
    topo_df  = pd.read_parquet(OUT_DIR / "topology_mapping.parquet")
    expected = set(TOPOLOGY_MAP.values())
    found    = set(topo_df["Topology_ID"].unique())
    missing  = expected - found
    print(f"\n  Topology IDs in mapping : {sorted(found)}")
    if missing:
        print(f"  WARNING — Topology IDs missing from mapping: {missing}")
    else:
        print(f"  All 8 topology IDs confirmed present.")

    return all_ok


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pipeline_start = time.time()

    print(f"\n{SEP}")
    print(f"  NexusWatch — Ingestion Pipeline")
    print(f"  Dataset  : HI-Medium")
    print(f"  Hardware : Xeon W-2245 ({CPU_CORES} threads) | 64 GB RAM | RTX A6000 48 GB")
    print(f"{SEP}")

    valid_ids = ingest_accounts()
    ingest_transactions(valid_ids)
    parse_topology_patterns()
    passed = validate_outputs()

    section("PIPELINE COMPLETE")
    total_time = time.time() - pipeline_start
    status     = "ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED — review above"

    print(f"""
  Status      : {status}
  Total time  : {total_time:.1f}s

  Output files in ./output/:
    accounts.parquet             <- clean account master
    transactions_clean.parquet   <- orphan-free, typed, time-featured
    topology_mapping.parquet     <- (From_Bank, To_Bank) -> Topology_ID

  Next: nexuswatch_graph_builder.py
  -----------------------------------------------------------------------
  Node features  : Entity Type (one-hot) | Bank ID (normalized)
  Edge features  : log(Amount) | Hour | DayOfWeek | Payment Format (OHE)
                   Topology_ID | FX mismatch flag (Recv != Pay currency)
  Node labels    : is_laundering (1 if ANY incident edge is fraud)
  Graph format   : torch_geometric.data.Data -> nexuswatch_graph.pt

  A6000 specific settings for graph builder:
    GraphSAGE neighbor sizes : [50, 25]  (vs default [25, 10])
    Training mode            : full-batch (entire graph fits in 48 GB VRAM)
    pos_weight               : set to imbalance ratio printed above
    torch.backends.cudnn.benchmark = True
  -----------------------------------------------------------------------
    """)