# %%
from pathlib import Path
from datetime import datetime
import time

import psutil
import pandas as pd


# %%

OUTPUT_FILE = Path("ram_usage.csv")
INTERVAL_SECONDS = 1

print(f"Monitoring RAM every {INTERVAL_SECONDS}s")
print(f"Saving to {OUTPUT_FILE.resolve()}")

while True:
        
    if not OUTPUT_FILE.exists():
        pd.DataFrame(
            columns=[
                "timestamp",
                "used_gb",
                "available_gb",
                "total_gb",
                "percent_used",
                "event",
            ]
        ).to_csv(OUTPUT_FILE, index=False)


    vm = psutil.virtual_memory()

    row = pd.DataFrame(
        {
            "timestamp": [datetime.now()],
            "used_gb": [vm.used / 1024**3],
            "available_gb": [vm.available / 1024**3],
            "total_gb": [vm.total / 1024**3],
            "percent_used": [vm.percent],
            "event": [None],
        }
    )

    row.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=False,
        index=False,
    )

    print(
        f"{datetime.now():%Y-%m-%d %H:%M:%S} | "
        f"RAM: {vm.percent:.1f}% "
        f"({vm.used / 1024**3:.1f} GB used)"
    )

    time.sleep(INTERVAL_SECONDS)


