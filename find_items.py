import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from PIL import Image
from collections import defaultdict
import config

img = np.array(Image.open(config.SCREENSHOTS_DIR / 'current_screen.png'))
r = img[:,:,0].astype(int)
g = img[:,:,1].astype(int)
b = img[:,:,2].astype(int)

# 狐狸橙色
fox_mask = (r > 200) & (g > 100) & (g < 155) & (b < 90) & (r - g > 80) & (r - b > 120)
ys, xs = np.where(fox_mask)
print(f"狐狸像素数: {len(xs)}")

cells = defaultdict(list)
for x, y in zip(xs, ys):
    cells[(x // 100, y // 100)].append((x, y))

print("\n狐狸位置（每格>300像素）:")
for (cx, cy), pts in sorted(cells.items()):
    if len(pts) > 300:
        mx = int(np.mean([p[0] for p in pts]))
        my = int(np.mean([p[1] for p in pts]))
        shelf = "右书架" if mx > 720 else "左书架"
        print(f"  ({mx}, {my})  像素={len(pts)}  [{shelf}]")

# 找空位：书架内部的浅米色区域（非物品）
# 书架内浅色背景约 RGB(230-245, 215-230, 190-210)
empty_mask = (r > 225) & (r < 250) & (g > 210) & (g < 235) & (b > 185) & (b < 215)
ey, ex = np.where(empty_mask)
print(f"\n空位像素数: {len(ex)}")
ecells = defaultdict(list)
for x, y in zip(ex, ey):
    ecells[(x // 100, y // 100)].append((x, y))

print("空位位置（每格>500像素）:")
for (cx, cy), pts in sorted(ecells.items()):
    if len(pts) > 500:
        mx = int(np.mean([p[0] for p in pts]))
        my = int(np.mean([p[1] for p in pts]))
        shelf = "右书架" if mx > 720 else "左书架"
        print(f"  ({mx}, {my})  像素={len(pts)}  [{shelf}]")
