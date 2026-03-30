import subprocess
import sys
import os

prefixes = ["HI-Small", "LI-Large", "LI-Medium", "LI-Small"]

for prefix in prefixes:
    print(f"\n==============================================")
    print(f"RUNNING PIPELINE FOR DATASET: {prefix}")
    print(f"==============================================\n")
    
    env = os.environ.copy()
    env["DATASET_PREFIX"] = prefix
    
    # Run the script
    result = subprocess.run(
        [sys.executable, "nexus_ingestion.py"],
        env=env,
        check=True
    )
    
    print(f"\n---> Finished {prefix}")
