import json
from pathlib import Path

RUNS_DIR = Path(".runs")

print(f"Checking runs in {RUNS_DIR.absolute()}...")

runs = sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()], reverse=True)
print(f"Found {len(runs)} runs.")

for i, run_dir in enumerate(runs[:15]):
    run_id = run_dir.name
    metrics_path = run_dir / "metrics.json"
    
    print(f"[{i}] {run_id}: ", end="")
    
    if not metrics_path.exists():
        print("NO METRICS FILE")
        continue
        
    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        status = metrics.get("status", "unknown")
        checkpoints = metrics.get("checkpoints", [])
        
        print(f"Status={status}, Checkpoints={len(checkpoints)}")
        if checkpoints:
            print(f"   -> Last ckpt: {checkpoints[-1]['path']}")
            
    except Exception as e:
        print(f"ERROR LOADING METRICS: {e}")

