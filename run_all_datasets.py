import subprocess
import sys
import os

# ALL datasets you showed
prefixes = [
    "HI-Small",
    "HI-Medium",
    "LI-Small",
    "LI-Medium",
    "LI-Large"
]

for prefix in prefixes:
    print(f"\n==============================================")
    print(f"🚀 RUNNING FULL PIPELINE FOR: {prefix}")
    print(f"==============================================\n")

    env = os.environ.copy()
    env["DATASET_PREFIX"] = prefix

    # 1. Ingestion
    subprocess.run(
        [sys.executable, "nexus_ingestion.py"],
        env=env,
        check=True
    )

    # 2. Graph Builder
    subprocess.run(
        [sys.executable, "nexuswatch_graph_builder.py"],
        env=env,
        check=True
    )

    # 3. Train Model
    subprocess.run(
        [sys.executable, "train_gnn.py"],
        env=env,
        check=True
    )

    print(f"\n✅ COMPLETED: {prefix}\n")