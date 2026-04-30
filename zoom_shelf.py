import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import config
from PIL import Image, ImageDraw
img = Image.open(config.SCREENSHOTS_DIR / 'current_screen.png')
# 裁剪书架区域并放大
shelf = img.crop((0, 900, 1080, 1320))
shelf = shelf.resize((1080*2, 420*2), Image.NEAREST)
draw = ImageDraw.Draw(shelf)
# 画竖线标注列位置（按120px间隔）
for x in range(0, 1080*2, 120*2):
    draw.line([(x, 0), (x, 840)], fill="red", width=2)
    draw.text((x+5, 5), str(x//2), fill="red")
shelf.save(config.SCREENSHOTS_DIR / 'shelf_zoom.png')
print("done")
