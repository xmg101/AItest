"""
分析当前关卡：区分书架结构和格子位置
用法：python analyze_level.py [截图文件名]（默认 current_state.png）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image, ImageDraw
import config

# ─── 读取截图 ──────────────────────────────────────────────────────────
fname = sys.argv[1] if len(sys.argv) > 1 else 'current_state.png'
img = Image.open(config.SCREENSHOTS_DIR / fname).convert('RGB')
W, H = img.size
arr = np.array(img)
r = arr[:,:,0].astype(int)
g = arr[:,:,1].astype(int)
b = arr[:,:,2].astype(int)
print(f"图像尺寸: {W}x{H}  文件: {fname}")

# ─── 检测木框像素（橙棕色） ─────────────────────────────────────────────
# 采样确认的木框颜色范围
wood = (
    # 深色木框边缘 (R≈165-215, G≈95-140, B≈40-80)
    ((r >= 155) & (r <= 225) & (g >= 85)  & (g <= 150) & (b >= 30) & (b <= 90) & (r - b > 90)) |
    # 中色木框面 (R≈195-225, G≈150-180, B≈95-135)
    ((r >= 190) & (r <= 235) & (g >= 145) & (g <= 185) & (b >= 85) & (b <= 140) & (r - b > 60))
)

print(f"木框总像素: {wood.sum()}")

# ─── 只在书架高度范围内处理 ─────────────────────────────────────────────
# 通过行投影确定书架 y 范围
row_wood = wood.sum(axis=1)
shelf_rows = np.where(row_wood > 30)[0]
if len(shelf_rows) == 0:
    print("未检测到书架木框，请检查截图")
    sys.exit(1)

y_top = int(shelf_rows[0])
y_bot = int(shelf_rows[-1])
print(f"书架 y 范围: {y_top} – {y_bot}  (高度 {y_bot-y_top}px)")

# ─── 在书架范围内找水平分隔线（行投影峰值） ────────────────────────────
# 用高斯平滑找行木框密度峰
row_density = wood[y_top:y_bot, :].sum(axis=1).astype(float)
# 简单平滑（5行滑动均值）
from numpy.lib.stride_tricks import sliding_window_view
smooth = np.convolve(row_density, np.ones(7)/7, mode='same')

# 找局部峰（比周围显著高）
peaks_y = []
for i in range(5, len(smooth)-5):
    if smooth[i] == smooth[i-5:i+6].max() and smooth[i] > 50:
        peaks_y.append(i + y_top)

# 合并相近峰（20px以内）
merged_y = []
for y in sorted(set(peaks_y)):
    if merged_y and y - merged_y[-1] < 20:
        merged_y[-1] = (merged_y[-1] + y) // 2
    else:
        merged_y.append(y)

print(f"水平木框线 (y): {merged_y}")

# ─── 找垂直分隔线（列投影峰值） ──────────────────────────────────────────
col_density = wood[y_top:y_bot, :].sum(axis=0).astype(float)
smooth_col = np.convolve(col_density, np.ones(7)/7, mode='same')

peaks_x = []
for i in range(5, len(smooth_col)-5):
    if smooth_col[i] == smooth_col[i-5:i+6].max() and smooth_col[i] > 30:
        peaks_x.append(i)

merged_x = []
for x in sorted(set(peaks_x)):
    if merged_x and x - merged_x[-1] < 20:
        merged_x[-1] = (merged_x[-1] + x) // 2
    else:
        merged_x.append(x)

print(f"垂直木框线 (x): {merged_x}")

# ─── 确定书架和格子边界 ─────────────────────────────────────────────────
# x 方向：分析列密度，找书架的起止和内部分隔
# 书架外部边界：列密度持续>0 的连续段
col_has_wood = (smooth_col > 15).astype(int)
col_diff = np.diff(np.concatenate([[0], col_has_wood, [0]]))
starts = np.where(col_diff == 1)[0]
ends   = np.where(col_diff == -1)[0]

shelf_x_ranges = [(int(s), int(e-1)) for s,e in zip(starts, ends) if e-s > 30]
print(f"\n书架 x 段: {shelf_x_ranges}")

# ─── 推断每个书架的格子 ──────────────────────────────────────────────────
# 用水平线把书架分成行，用垂直线把每行分成列
# 先把水平线和垂直线整理成网格边界

# 边界 y 列表（加上 y_top/y_bot）
h_bounds = sorted(set([y_top] + merged_y + [y_bot]))
# 边界 x 列表
v_bounds = sorted(set(merged_x))

print(f"\n最终水平边界 y: {h_bounds}")
print(f"最终垂直边界 x: {v_bounds}")

# ─── 为每个书架区段生成格子 ──────────────────────────────────────────────
cells = []
shelf_id = 0

for sx, ex in shelf_x_ranges:
    shelf_id += 1
    shelf_label = chr(ord('A') + shelf_id - 1)  # A, B, C...

    # 找属于这个书架的 x 边界（在 sx..ex 范围内的竖线）
    inner_x = [sx] + [x for x in v_bounds if sx + 20 < x < ex - 20] + [ex]
    inner_x = sorted(set(inner_x))

    # 找这个书架覆盖的 y 边界
    # 检测该书架 x 范围内是否有木框存在于每一个水平分段
    shelf_h_bounds = [y_top]
    for y in merged_y:
        # 检查该 y 处在该 x 范围内是否有木框
        wood_at_y = wood[max(0,y-5):min(H,y+5), sx:ex].sum()
        if wood_at_y > 20:
            shelf_h_bounds.append(y)
    shelf_h_bounds.append(y_bot)
    shelf_h_bounds = sorted(set(shelf_h_bounds))

    print(f"\n书架{shelf_label} (x={sx}-{ex}):")
    print(f"  x 边界: {inner_x}")
    print(f"  y 边界: {shelf_h_bounds}")

    row_idx = 0
    for i in range(len(shelf_h_bounds) - 1):
        y1_c = shelf_h_bounds[i]
        y2_c = shelf_h_bounds[i + 1]
        if y2_c - y1_c < 30:
            continue
        row_idx += 1

        col_idx = 0
        for j in range(len(inner_x) - 1):
            x1_c = inner_x[j]
            x2_c = inner_x[j + 1]
            if x2_c - x1_c < 30:
                continue
            col_idx += 1

            # 验证该格子内部确实有浅色（排除纯木框区域）
            cell_region = arr[y1_c+10:y2_c-10, x1_c+10:x2_c-10]
            if cell_region.size == 0:
                continue

            cx = (x1_c + x2_c) // 2
            cy = (y1_c + y2_c) // 2
            cells.append({
                'shelf': shelf_label,
                'row': row_idx,
                'col': col_idx,
                'bbox': (x1_c, y1_c, x2_c, y2_c),
                'center': (cx, cy),
            })
            print(f"  格子 {shelf_label}{row_idx}{col_idx}: "
                  f"bbox=({x1_c},{y1_c})-({x2_c},{y2_c})  中心=({cx},{cy})")

# ─── 绘制标注图 ────────────────────────────────────────────────────────
out = img.copy()
draw = ImageDraw.Draw(out)

SHELF_COLORS = ['#00AA44', '#DD2222', '#2255FF', '#FF8800']
for c in cells:
    idx = ord(c['shelf']) - ord('A')
    color = SHELF_COLORS[idx % len(SHELF_COLORS)]
    x1, y1, x2, y2 = c['bbox']
    # 格子边框
    draw.rectangle([x1+4, y1+4, x2-4, y2-4], outline=color, width=4)
    # 中心十字
    cx, cy = c['center']
    draw.line([(cx-15,cy),(cx+15,cy)], fill=color, width=3)
    draw.line([(cx,cy-15),(cx,cy+15)], fill=color, width=3)
    # 标签
    label = f"{c['shelf']}{c['row']}{c['col']}"
    draw.rectangle([x1+5, y1+5, x1+48, y1+30], fill=color)
    draw.text((x1+7, y1+7), label, fill='white')

# 书架大框
for i, (sx, ex) in enumerate(shelf_x_ranges):
    color = SHELF_COLORS[i % len(SHELF_COLORS)]
    draw.rectangle([sx+2, y_top+2, ex-2, y_bot-2], outline=color, width=3)
    draw.text((sx+6, y_top+6), f"书架{'ABCDE'[i]}", fill=color)

out.save(config.SCREENSHOTS_DIR / 'level_analyzed.png')
print(f"\n标注图: screenshots/level_analyzed.png")
print(f"共识别 {len(cells)} 个格子，{len(shelf_x_ranges)} 个书架")
