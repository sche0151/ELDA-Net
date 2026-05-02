import cv2
import numpy as np
from pathlib import Path

culane_root = Path("data/culane_raw")
out_img_dir = Path("data/culane/images")
out_label_dir = Path("data/culane/labels")

out_img_dir.mkdir(parents=True, exist_ok=True)
out_label_dir.mkdir(parents=True, exist_ok=True)

for img_path in culane_root.rglob("*.jpg"):
    label_txt = img_path.with_suffix(".lines.txt")
    if not label_txt.exists():
        continue

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    with open(label_txt, "r") as f:
        for line in f:
            nums = list(map(float, line.strip().split()))
            points = [(int(nums[i]), int(nums[i + 1])) for i in range(0, len(nums), 2)]

            for p1, p2 in zip(points[:-1], points[1:]):
                cv2.line(mask, p1, p2, 255, thickness=5)

    out_name = img_path.name
    cv2.imwrite(str(out_img_dir / out_name), img)
    cv2.imwrite(str(out_label_dir / out_name), mask)