import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import config
from PIL import Image, ImageDraw
img = Image.open(config.SCREENSHOTS_DIR / 'current_screen.png').copy()
draw = ImageDraw.Draw(img)

# 根据放大图重新标注 y≈1215，x按格子分布
points = [
    # 左书架底行各格 (每格约180px宽，3格 x=60,240,420,600)
    (60,  1215, "左1-fox", "lime"),
    (240, 1215, "左2",     "lime"),
    (420, 1215, "左3",     "lime"),
    (600, 1215, "左4",     "lime"),
    # 右书架底行各格
    (810, 1215, "右1",   "red"),
    (960, 1215, "右2",   "red"),
    (1020,1215, "右fox", "yellow"),
]
r = 22
for x, y, label, color in points:
    draw.ellipse([x-r, y-r, x+r, y+r], outline=color, width=5)
    draw.text((x-20, y+28), label, fill=color)
img.save(config.SCREENSHOTS_DIR / 'marked.png')
print("done")
