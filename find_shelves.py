"""
find_shelves.py — 检测书架内边缘轮廓

3D 等角渲染规律：
  横板结构（截面从上往下）：背景 → 板体 → 亮面(底部装饰高光) → 背景
  竖隔板/外壁（截面从内往外）：内容区 → 亮面(朝内高光) → 暗影(朝外阴影) → 板体

内边缘定义：
  水平上边界(ey1) = 跳过横板底部亮面后第一个真实内容行
  水平下边界(ey2) = 下方横板体最后一行（亮面之前）
  垂直左边界(cx1) = 亮面右边界+1（朝右入射的亮面穿越后第一内容列）
  垂直右边界(cx2) = 亮面左边界（朝左入射的亮面左缘 = 内容区终止列）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label as sp_label
import config

# ── 读图 ──────────────────────────────────────────────────────────────────────
fname = sys.argv[1] if len(sys.argv) > 1 else "current_state.png"
img = Image.open(config.SCREENSHOTS_DIR / fname).convert("RGB")
W, H = img.size
arr = np.array(img)
r = arr[:, :, 0].astype(int)
g = arr[:, :, 1].astype(int)
b = arr[:, :, 2].astype(int)
print(f"图像: {W}×{H}")

# ── 掩码 ──────────────────────────────────────────────────────────────────────
# 板芯颜色 R≈196 G≈117 B≈57；G<110 的阴影行被排除
board = (
    (r >= 185) & (r <= 215) &
    (g >= 110) & (g <= 132) &
    (b >= 48)  & (b <= 70)
)
ui_top = int(H * 0.12)
ui_bot = int(H * 0.78)
board[:ui_top, :] = False
board[ui_bot:,  :] = False

BG = np.array([239, 215, 187], dtype=float)
is_bg   = np.abs(arr.astype(float) - BG).max(axis=-1) < 35
is_item = (~is_bg) & (~board)

# ── 工具：连续段检测 ───────────────────────────────────────────────────────────
def spans(mask, min_len=40, gap_fill=15):
    a = mask.astype(int).copy()
    i = 0
    while i < len(a):
        if a[i] == 0:
            j = i
            while j < len(a) and a[j] == 0:
                j += 1
            if j - i <= gap_fill:
                a[i:j] = 1
            i = max(i + 1, j)
        else:
            i += 1
    segs, in_s, start = [], False, 0
    for i, v in enumerate(a):
        if v and not in_s:
            in_s, start = True, i
        elif not v and in_s:
            in_s = False
            if i - start >= min_len:
                segs.append((start, i))
    if in_s and len(a) - start >= min_len:
        segs.append((start, len(a)))
    return segs

def band_x_segs(brd, y1, y2, min_len=50, gap_fill=20):
    return spans(brd[y1:y2].any(axis=0), min_len=min_len, gap_fill=gap_fill)

def x_overlaps(cx1, cx2, x_segs, min_overlap=30):
    return any(min(cx2, ux2) - max(cx1, ux1) >= min_overlap for ux1, ux2 in x_segs)

# ── 水平边界：亮面行检测 ───────────────────────────────────────────────────────
def first_content_row(a, y_start, max_scan=50):
    """向下跳过横板底部亮面（R>240, R-B>90 占行宽>20%），返回第一真实内容行。"""
    W_ = a.shape[1]
    for y in range(y_start, min(a.shape[0], y_start + max_scan)):
        rv = a[y, :, 0].astype(int)
        bv = a[y, :, 2].astype(int)
        if int(((rv > 240) & ((rv - bv) > 90)).sum()) < W_ * 0.20:
            return y
    return y_start + max_scan

# ── Step 1: 找所有水平横板带 ──────────────────────────────────────────────────
row_counts = board.sum(axis=1).astype(float)
dense_mask = (row_counts > 150).astype(int)
dense_mask[:ui_top] = 0
dense_mask[ui_bot:] = 0
labeled, n_bands = sp_label(dense_mask)

all_bands = []
for bi in range(1, n_bands + 1):
    ys = np.where(labeled == bi)[0]
    if len(ys) < 4:
        continue
    all_bands.append((int(ys[0]), int(ys[-1])))
all_bands.sort()

print(f"\n找到 {len(all_bands)} 条横板:")
for i, (y1, y2) in enumerate(all_bands):
    print(f"  横板{i}: y={y1}–{y2}")

if len(all_bands) < 2:
    print("横板不足，退出")
    sys.exit(1)

# ── Step 2: 顶板 x 段 → 书架候选列 ───────────────────────────────────────────
top_xs = band_x_segs(board, *all_bands[0])
print(f"\n顶板 x 段: {top_xs}")

x_edges = sorted(set(x for seg in top_xs for x in seg))

# ── 左/右外壁内边缘：统一用亮面边界扫描 ──────────────────────────────────────
def detect_wall_inner_edges(a, h, w):
    """
    左壁：右向扫找最后一个亮面列 +1（= 左壁内边缘 = 第一内容列）
    右壁：左向扫找亮面左边界（= 右壁内边缘 = 内容区最右列的下一列）
    两侧均以亮面（R>245, R-B>100）为依据，保持检测逻辑一致。
    """
    rv = a[:, :, 0].astype(int)
    bv = a[:, :, 2].astype(int)
    content_ys = range(int(h * 0.35), int(h * 0.65), 20)
    lefts, rights = [], []
    for y in content_ys:
        # 左壁：找最右亮面列 +1
        li = 0
        for x in range(w // 4):
            if rv[y, x] > 245 and (rv[y, x] - bv[y, x]) > 100:
                li = x + 1
        lefts.append(li)
        # 右壁：右→左扫，进入亮面后找其左边界
        ri = w
        in_bright = False
        for x in range(w - 1, w * 3 // 4, -1):
            if rv[y, x] > 245 and (rv[y, x] - bv[y, x]) > 100:
                ri = x
                in_bright = True
            elif in_bright:
                break  # 已越过亮面左边界
        rights.append(ri)
    li_med = int(np.median(lefts)) if lefts else 0
    ri_med = int(np.median(rights)) if rights else w
    if li_med >= w // 4:
        li_med = (x_edges[0] + 20) if x_edges else 0
    if ri_med >= w:
        ri_med = (x_edges[-1] - 20) if x_edges else w
    return li_med, ri_med

left_inner, right_inner = detect_wall_inner_edges(arr, H, W)
print(f"外壁内边缘: left={left_inner}, right={right_inner}")

outer_left  = x_edges[0]  if x_edges else 0
outer_right = x_edges[-1] if x_edges else W
x_edges_adj = [left_inner  if x == outer_left  else
               right_inner if x == outer_right else x
               for x in x_edges]
all_x_bounds = [0] + x_edges_adj + [W]
x_candidates = [(a, b) for a, b in zip(all_x_bounds, all_x_bounds[1:]) if b - a >= 60]
print(f"x 候选列段: {x_candidates}")

# ── 中间竖隔板精确内边缘调整 ──────────────────────────────────────────────────
if len(top_xs) >= 2:
    gap_left_xs  = {seg[1] for seg in top_xs[:-1]}  # 左段右边缘（用作 cx2）
    gap_right_xs = {seg[0] for seg in top_xs[1:]}   # 右段左边缘（用作 cx1）
else:
    gap_left_xs, gap_right_xs = set(), set()

# 多行采样：上层书架内容区，4 个等距 y（提高鲁棒性，避免单行被物品遮挡）
_shelf_y1 = first_content_row(arr, all_bands[0][1] + 1)
_shelf_y2 = all_bands[1][0]
_scan_ys = [int(_shelf_y1 + (_shelf_y2 - _shelf_y1) * f) for f in (0.25, 0.40, 0.55, 0.70)]

def _bright_left(x, y, max_skip=20, max_face=80):
    """向左找亮面（R>245, R-B>100）的左边界。cx2 调整用。"""
    rv_ = arr[y, :, 0].astype(int)
    bv_ = arr[y, :, 2].astype(int)
    xi = x - 1
    while xi >= x - max_skip and not (rv_[xi] > 245 and rv_[xi] - bv_[xi] > 100):
        xi -= 1
    if xi < x - max_skip:
        return None
    while xi >= 0 and rv_[xi] > 245 and rv_[xi] - bv_[xi] > 100:
        xi -= 1
    return xi + 1

def _bright_right_end(x, y, max_skip=15, max_face=80):
    """向右跳过亮面（R>240, R-B>80），返回第一内容列。cx1 调整用。"""
    rv_ = arr[y, :, 0].astype(int)
    bv_ = arr[y, :, 2].astype(int)
    xi = x
    while xi < x + max_skip and not (rv_[xi] > 240 and rv_[xi] - bv_[xi] > 80):
        xi += 1
    while xi < W and rv_[xi] > 240 and rv_[xi] - bv_[xi] > 80:
        xi += 1
    return xi

def _dark_left(x, y, max_scan=12):
    """向左找暗影（R<190, G<115, R-B>50）最左像素。cx1 调整用（S3左）。"""
    rv_ = arr[y, :, 0].astype(int)
    gv_ = arr[y, :, 1].astype(int)
    bv_ = arr[y, :, 2].astype(int)
    xi = x - 1
    while xi >= max(0, x - max_scan) and rv_[xi] < 190 and gv_[xi] < 115 and rv_[xi] - bv_[xi] > 50:
        xi -= 1
    return xi + 1

def _dark_right_end(x, y, max_scan=12):
    """向右找暗影末端+1。cx2 调整用（S3右）。"""
    rv_ = arr[y, :, 0].astype(int)
    gv_ = arr[y, :, 1].astype(int)
    bv_ = arr[y, :, 2].astype(int)
    xi = x
    while xi < min(W, x + max_scan) and rv_[xi] < 190 and gv_[xi] < 115 and rv_[xi] - bv_[xi] > 50:
        xi += 1
    return xi

def _median_adj(fn, x):
    """多行采样取中位数，过滤 None。"""
    vals = [v for y in _scan_ys for v in [fn(x, y)] if v is not None]
    return int(np.median(vals)) if vals else x

# cx1 调整表
gap_cx1_adj = {}
for _x in gap_right_xs:  # 右侧竖隔板左边缘 → cx1：跳过暗影+亮面，取第一内容列
    gap_cx1_adj[_x] = _median_adj(_bright_right_end, _x)
for _x in gap_left_xs:   # 左侧竖隔板右边缘 → cx1（S3左）：取暗影最左列
    gap_cx1_adj[_x] = _median_adj(_dark_left, _x)

# 隔板厚度参考（从 cx1 调整量反推，用于 cx2 回退）
_div_thickness = (
    int(np.mean([gap_cx1_adj[k] - k for k in gap_right_xs]))
    if gap_right_xs else 22
)

# cx2 调整表
gap_cx2_adj = {}
for _x in gap_left_xs:   # 左侧竖隔板右边缘 → cx2（S0/S2右）：找亮面左边界
    vals = [_bright_left(_x, y) for y in _scan_ys]
    vals = [v for v in vals if v is not None]
    gap_cx2_adj[_x] = int(np.median(vals)) if vals else _x - _div_thickness
for _x in gap_right_xs:  # 右侧竖隔板左边缘 → cx2（S3右）：暗影末端
    gap_cx2_adj[_x] = _median_adj(_dark_right_end, _x)

print(f"cx1 调整: {gap_cx1_adj}")
print(f"cx2 调整: {gap_cx2_adj}")

# ── Step 3: 行带 × x 候选 → 书架 ─────────────────────────────────────────────
shelves = []

for i in range(len(all_bands) - 1):
    row_y1 = all_bands[i][1]
    row_y2 = all_bands[i + 1][0]
    if row_y2 - row_y1 < 50:
        continue

    upper_xs = band_x_segs(board, *all_bands[i])
    print(f"\n行带 y={row_y1}–{row_y2}  上横板x段={upper_xs}:")

    for cx1, cx2 in x_candidates:
        if not x_overlaps(cx1, cx2, upper_xs):
            print(f"  x={cx1}–{cx2}: 上横板无覆盖 → 跳过")
            continue

        cx1e = gap_cx1_adj.get(cx1, cx1)
        cx2e = gap_cx2_adj.get(cx2, cx2)

        item_frac = is_item[row_y1:row_y2, cx1e:cx2e].mean()
        print(f"  x={cx1e}–{cx2e}: 物品={item_frac:.3f}", end="")
        if item_frac < 0.05:
            print("  → 空，跳过")
            continue

        # ey1：横板底部亮面后第一个真实内容行
        ey1 = first_content_row(arr, all_bands[i][1] + 1)

        # ey2：下方横板体最后一行（亮面之前，包含等角视角下物品覆盖的板面区域）
        ey2 = all_bands[i + 1][1]

        # x 边界收紧到实际物品范围
        xs_c = np.where(is_item[row_y1:row_y2, cx1e:cx2e].any(axis=0))[0]
        ex1 = (cx1e + int(xs_c[0]))      if len(xs_c) else cx1e
        ex2 = (cx1e + int(xs_c[-1]) + 1) if len(xs_c) else cx2e

        sid = len(shelves)
        shelves.append({"id": sid, "x1": ex1, "x2": ex2, "y1": ey1, "y2": ey2})
        print(f"  → 书架{sid}: x={ex1}–{ex2}, y={ey1}–{ey2}"
              f"  ({ex2 - ex1}×{ey2 - ey1})")

print(f"\n共检测到 {len(shelves)} 个书架")

# ── Step 4: 画轮廓 ────────────────────────────────────────────────────────────
COLORS = ["#FF4444", "#4488FF", "#44CC66", "#FF9900",
          "#FF44FF", "#00FFCC", "#FFFF44", "#FF8844"]
out = img.copy()
draw = ImageDraw.Draw(out)

_font = None
for _fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc"]:
    try:
        _font = ImageFont.truetype(_fp, 22)
        break
    except Exception:
        pass

for s in shelves:
    c = COLORS[s["id"] % len(COLORS)]
    draw.rectangle([s["x1"], s["y1"], s["x2"], s["y2"]], outline=c, width=6)
    draw.rectangle([s["x1"], s["y1"], s["x1"] + 160, s["y1"] + 36], fill=c)
    draw.text((s["x1"] + 5, s["y1"] + 5),
              f"书架{s['id']}  {s['x2'] - s['x1']}×{s['y2'] - s['y1']}",
              fill="white", font=_font)

for y1, y2 in all_bands:
    draw.line([(0, (y1 + y2) // 2), (W, (y1 + y2) // 2)], fill="#00FF00", width=2)

out.save(config.SCREENSHOTS_DIR / "shelves_found.png")
print(f"\n结果已保存: screenshots/shelves_found.png  (绿线=横板中心)")
