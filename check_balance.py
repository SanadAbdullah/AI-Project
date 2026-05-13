from pathlib import Path
from collections import Counter

labels_dir = Path("yolo_dataset/train/labels")
counts = Counter()
for lf in labels_dir.glob("*.txt"):
    for line in lf.read_text().strip().splitlines():
        if line:
            counts[int(line.split()[0])] += 1

names = {0:"safety_helmet", 1:"fire_extinguisher", 2:"emergency_exit", 3:"traffic_cone"}
for idx, name in names.items():
    print(f"{name}: {counts[idx]} instances")