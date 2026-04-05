def ingest_transactions(valid_ids: frozenset) -> int:
    section("TASK 2 — Transactions Ingestion (Chunked to prevent RAM Crash)")
    t0 = time.time()

    print(f"\n  Loading CSV in chunks...")
    chunk_size = 2_000_000
    df_list = []
    
    # Process the massive CSV in chunks to save memory
    for i, chunk in enumerate(pd.read_csv(TRANS_FILE, dtype=str, chunksize=chunk_size)):
        chunk = _fix_duplicate_account_columns(chunk)
        chunk["From Account"] = chunk["From Account"].str.strip().str.upper()
        chunk["To Account"]   = chunk["To Account"].str.strip().str.upper()
        
        chunk.dropna(subset=["From Account", "To Account"], inplace=True)
        
        # Orphan drop on just this chunk
        mask = chunk["From Account"].isin(valid_ids) & chunk["To Account"].isin(valid_ids)
        clean_chunk = chunk[mask].copy()
        df_list.append(clean_chunk)
        print(f"    -> Processed chunk {i+1}...")

    df = pd.concat(df_list, ignore_index=True)
    raw_count = len(df)
    print(f"  Clean rows combined: {raw_count:>12,} ({elapsed(t0)})")

    # Step 6: Type casting
    print(f"\n  Casting types...")
    df["Amount Paid"]     = pd.to_numeric(df["Amount Paid"], errors="coerce")
    df["Amount Received"] = pd.to_numeric(df["Amount Received"], errors="coerce")
    df["Is Laundering"]   = pd.to_numeric(df["Is Laundering"], errors="coerce").astype("Int8")
    df["Timestamp"]       = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Step 7: Derive time features
    df["Hour"]      = df["Timestamp"].dt.hour.astype("Int8")
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek.astype("Int8")
    df["Month"]     = df["Timestamp"].dt.month.astype("Int8")

    print(f"\n  Saving to Parquet (Snappy)...")
    out_path = OUT_DIR / "transactions_clean.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    
    return len(df)