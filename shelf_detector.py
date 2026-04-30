"""
shelf_detector.py — 从游戏截图中检测货架格子位置

检测思路：
  1. 扫描全图找出书架内容的 y 区间
  2. 在此区间内找行分隔线 → 确定多个水平行带
  3. 每个行带内，用大 gap_fill 找书架单元（x 簇）
  4. 每个书架单元内，用小 gap_fill 找格子（列簇）
  5. 按行/位置顺序分配 shelf_id
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image, ImageDraw
from dataclasses import dataclass
from typing import Optional
import config

# ── 颜色常量 ──────────────────────────────────────────────────────────────────

BG    = np.array([239, 215, 187], dtype=float)
BG_TOL = 32

def is_bg(px: np.ndarray) -> np.ndarray:
    return np.abs(px.astype(float) - BG).max(axis=-1) < BG_TOL

def is_wood(px: np.ndarray) -> np.ndarray:
    r = px[..., 0].astype(int)
    g = px[..., 1].astype(int)
    b = px[..., 2].astype(int)
    bright = (r - b > 55) & (r > g) & (g > b) & (r > 100)
    dark   = (r < 110) & (g < 85) & (b < 45) & (r > 20) & ((r - b) > 10)
    return bright | dark

# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class Slot:
    shelf_id: int   # 书架编号（按行从左到右、行从上到下递增）
    row: int        # 书架内行号（普通书架均为 0）
    col: int        # 书架内列号
    cx: int         # 屏幕 x
    cy: int         # 屏幕 y
    w: int
    h: int

    def __repr__(self):
        return (f"Slot(S{self.shelf_id} r{self.row}c{self.col} "
                f"@ ({self.cx},{self.cy}) {self.w}×{self.h})")

# ── 主检测器 ──────────────────────────────────────────────────────────────────

class ShelfDetector:

    def __init__(self, img: np.ndarray):
        self.img = img
        self.h, self.w = img.shape[:2]

    def detect(self) -> list[Slot]:
        """返回所有格子的屏幕坐标列表。"""
        sy1, sy2 = self._shelf_y_range()
        row_bands = self._find_row_bands(sy1, sy2)

        # 全局参考（第一行顶板），用于无法直接扫到顶板时的回退
        global_ref = self._units_from_top_board(row_bands[0][0])

        slots = []
        shelf_id = 0

        for r_idx, (ry1, ry2) in enumerate(row_bands):
            # 优先：直接扫本行顶板，得到该行所有书架的精确 x 范围
            row_ref = self._units_from_top_board(ry1)
            if len(row_ref) >= 2:
                unit_spans = sorted(row_ref)
            else:
                # 回退：物品检测 + 全局参考修正
                unit_spans = self._find_shelf_units(ry1, ry2)
                unit_spans = self._apply_ref_units(unit_spans, global_ref)

            for ux1, ux2 in unit_spans:
                col_clusters = self._find_cols_by_dividers(ry1, ry2, ux1, ux2)
                rh = ry2 - ry1
                cy = int(ry1 + rh * 0.57)   # 格子内容区偏下，57% 处更贴近视觉中心
                for c_idx, (cx1, cx2) in enumerate(col_clusters):
                    slots.append(Slot(
                        shelf_id=shelf_id, row=0, col=c_idx,
                        cx=(cx1 + cx2) // 2, cy=cy,
                        w=cx2 - cx1, h=rh))
                shelf_id += 1

        return slots

    # ── 书架 y 范围 ───────────────────────────────────────────────────────────

    def _shelf_y_range(self) -> tuple[int, int]:
        non_bg = (~is_bg(self.img)).sum(axis=1)
        active = _spans(non_bg > self.w * 0.10, min_len=60, gap_fill=15)
        game_spans = [s for s in active
                      if s[0] > 200 and s[1] < int(self.h * 0.70)]
        if not game_spans:
            return (int(self.h * 0.30), int(self.h * 0.65))
        return min(s[0] for s in game_spans), max(s[1] for s in game_spans)

    # ── 行分隔 ────────────────────────────────────────────────────────────────

    def _find_row_bands(self, sy1: int, sy2: int) -> list[tuple[int, int]]:
        region = self.img[sy1:sy2]
        rh = sy2 - sy1
        wood_frac = is_wood(region).sum(axis=1) / self.w
        dividers = _spans(wood_frac > 0.05, min_len=4, gap_fill=3)

        boundaries = [0]
        for d0, d1 in dividers:
            boundaries.append((d0 + d1) // 2)
        boundaries.append(rh)

        rows = []
        for i in range(len(boundaries) - 1):
            a, b = boundaries[i], boundaries[i + 1]
            if b - a >= 40:
                pad = min(6, (b - a) // 8)
                rows.append((sy1 + a + pad, sy1 + b - pad))
        return rows if rows else [(sy1, sy2)]

    # ── 书架单元检测（粗）────────────────────────────────────────────────────

    def _find_shelf_units(self, ry1: int, ry2: int) -> list[tuple[int, int]]:
        """
        在行带内找书架单元的 x 范围。
        两步法：
          1. 找原始物品 x 簇（不填间隔），过滤 <15px 的单像素噪声
          2. 合并间距 ≤45px 的相邻簇（物品间距），>45px 的断开（书架间空隙）
        """
        region = self.img[ry1:ry2]
        item = (~is_bg(region)) & (~is_wood(region))
        per_col = item.sum(axis=0)
        row_h = ry2 - ry1
        thr = row_h * 0.08

        # Step 1: 原始簇，不填间隔，去掉噪声小片 (<15px)
        raw = _spans(per_col > thr, min_len=15, gap_fill=0)
        if not raw:
            return []

        # Step 2: 合并书架单元内部间隔（≤45px），保留书架间空隙（>45px）
        merged: list[list[int]] = [list(raw[0])]
        for x1, x2 in raw[1:]:
            if x1 - merged[-1][1] <= 45:
                merged[-1][1] = x2
            else:
                merged.append([x1, x2])

        # 过滤掉宽度 <60px 的残留噪声
        return [(x1, x2) for x1, x2 in merged if x2 - x1 >= 60]

    # ── 顶板参考边界 ──────────────────────────────────────────────────────────

    def _units_from_top_board(self, ry1: int) -> list[tuple[int, int]]:
        """
        扫描行带起始处（顶板木框所在位置）确定书架单元精确 x 边界。
        顶板在 ry1 起始的前 25px 内，比物品位置更可靠。
        """
        y1 = ry1
        y2 = ry1 + 25
        board = self.img[y1:y2]
        wood_col = is_wood(board).sum(axis=0)
        thr = (y2 - y1) * 0.5
        return _spans(wood_col > thr, min_len=60, gap_fill=10)

    def _apply_ref_units(self,
                         unit_spans: list[tuple[int, int]],
                         ref_units: list[tuple[int, int]]
                         ) -> list[tuple[int, int]]:
        """
        将检测到的书架单元 x 范围与顶板参考边界对齐：
        - 若某单元与参考单元重叠 >50%：替换为参考边界（精确宽度）
        - 若某单元落在两个参考单元之间的间隙里：替换为该间隙范围
        """
        if not ref_units:
            return unit_spans

        # 构建"间隙区间"：相邻两个参考单元之间的背景区
        gaps: list[tuple[int, int]] = []
        sorted_refs = sorted(ref_units)
        for i in range(len(sorted_refs) - 1):
            gaps.append((sorted_refs[i][1], sorted_refs[i + 1][0]))

        result = []
        for ux1, ux2 in unit_spans:
            uw = ux2 - ux1
            # 先尝试匹配参考单元
            matched = None
            for rx1, rx2 in ref_units:
                overlap = max(0, min(ux2, rx2) - max(ux1, rx1))
                if overlap / max(uw, 1) > 0.5:
                    matched = (rx1, rx2)
                    break
            if matched:
                result.append(matched)
                continue
            # 再尝试匹配间隙区间（中间书架）
            ucx = (ux1 + ux2) / 2
            gap_matched = None
            for gx1, gx2 in gaps:
                if gx1 <= ucx <= gx2:
                    gap_matched = (gx1, gx2)
                    break
            result.append(gap_matched if gap_matched else (ux1, ux2))
        return result

    def _item_cy(self, ry1: int, ry2: int, ux1: int, ux2: int) -> int:
        """
        用行内物品的 y 重心作为格子中心 y，比行带中点更准确。
        """
        region = self.img[ry1:ry2, ux1:ux2]
        item = (~is_bg(region)) & (~is_wood(region))
        per_row = item.sum(axis=1)   # shape (row_h,)
        total = per_row.sum()
        if total == 0:
            return (ry1 + ry2) // 2
        ys = np.arange(ry1, ry2)
        return int((ys * per_row).sum() / total)

    # ── 格子列检测（木质竖隔板 + 宽度回退）─────────────────────────────────────

    def _find_cols_by_dividers(self, ry1: int, ry2: int,
                               ux1: int, ux2: int) -> list[tuple[int, int]]:
        """
        检测书架单元内部的格子列数，支持 1/2/3/N 格任意数量。

        流程：
        1. 屏蔽横板行（避免横板信号淹没竖隔板信号）
        2. 用剩余行的木框列密度找竖隔板
        3. 若某区间宽度 > 1.5 × 预期格宽，则按预期格宽再均分
           （处理竖隔板被物品遮住检测不到的情况）
        """
        region = self.img[ry1:ry2, ux1:ux2]
        rh = ry2 - ry1
        unit_w = ux2 - ux1

        # 预期格子宽度 ≈ 行高 × 0.43（实测此游戏格宽略小于行高）
        typical_w = max(70, int(rh * 0.43))

        # ── Step 1: 屏蔽横板行 ────────────────────────────────────────────────
        wood_mask = is_wood(region)
        row_wood  = wood_mask.sum(axis=1)
        horiz     = row_wood > unit_w * 0.40   # 行内 >40% 为木框 → 横板
        wood_v    = wood_mask.copy()
        wood_v[horiz] = False
        remain_h  = int((~horiz).sum())

        if remain_h < 15:
            # 行高太小，直接按宽度均分
            n = max(1, round(unit_w / typical_w))
            return _divide_equal(ux1, ux2, n)

        # ── Step 2: 竖隔板列密度 ─────────────────────────────────────────────
        col_density = wood_v.sum(axis=0)
        # 35% 阈值 + min_len=8 双重过滤：真正竖隔板宽 ≥10px 且密度高，物品边缘窄且弱
        thr = remain_h * 0.35

        divider_spans = _spans(col_density > thr, min_len=8, gap_fill=4)
        # 保留真正竖隔板：离边缘 >45px，且宽度 ≤25px（物品/阴影造成的假峰更宽）
        inner = [(d0, d1) for d0, d1 in divider_spans
                 if d0 > 45 and d1 < unit_w - 45 and (d1 - d0) <= 25]

        # ── Step 3: 建立初始边界 ─────────────────────────────────────────────
        boundaries = [0] + [(d0 + d1) // 2 for d0, d1 in inner] + [unit_w]

        # ── Step 4: 宽区间再均分（补充被物品遮住的竖隔板）─────────────────────
        final_bounds: list[int] = []
        for i in range(len(boundaries) - 1):
            a, b = boundaries[i], boundaries[i + 1]
            w = b - a
            n_fit = max(1, round(w / typical_w))
            if n_fit <= 1:
                if not final_bounds:
                    final_bounds.append(a)
                final_bounds.append(b)
            else:
                step = w / n_fit
                for k in range(n_fit + 1):
                    v = int(a + k * step)
                    if not final_bounds or v > final_bounds[-1] + 20:
                        final_bounds.append(v)

        slots = [(ux1 + final_bounds[i], ux1 + final_bounds[i + 1])
                 for i in range(len(final_bounds) - 1)
                 if final_bounds[i + 1] - final_bounds[i] >= 40]

        return slots if slots else [(ux1, ux2)]

    # ── 格子列检测（细，小 gap_fill）──────────────────────────────────────────

    def _find_cols_in_unit(self, ry1: int, ry2: int,
                           ux1: int, ux2: int) -> list[tuple[int, int]]:
        """
        在书架单元 [ux1, ux2] 内找格子列。
        返回全局 x 坐标的 (x1, x2) 列表。
        """
        region = self.img[ry1:ry2, ux1:ux2]
        item = (~is_bg(region)) & (~is_wood(region))
        per_col = item.sum(axis=0)
        row_h = ry2 - ry1
        thr = row_h * 0.08

        local_clusters = _spans(per_col > thr, min_len=28, gap_fill=3)
        return [(ux1 + lx1, ux1 + lx2) for lx1, lx2 in local_clusters]

    # ── 可视化 ────────────────────────────────────────────────────────────────

    def visualize(self, slots: list[Slot],
                  out_path: Optional[Path] = None) -> Path:
        vis = Image.fromarray(self.img).copy()
        draw = ImageDraw.Draw(vis)
        palette = ["#FF4444", "#44AAFF", "#44FF88", "#FFAA00",
                   "#FF44FF", "#00FFFF", "#FFFF44", "#FF8844"]

        for slot in slots:
            c = palette[slot.shelf_id % len(palette)]
            r = 35
            draw.ellipse([slot.cx - r, slot.cy - r,
                          slot.cx + r, slot.cy + r],
                         outline=c, width=5)
            draw.text((slot.cx - 22, slot.cy - 10),
                      f"S{slot.shelf_id} c{slot.col}", fill=c)

        if out_path is None:
            out_path = config.SCREENSHOTS_DIR / "shelf_detected.png"
        vis.save(out_path)
        return out_path


# ── 工具 ──────────────────────────────────────────────────────────────────────

def _spans(mask: np.ndarray, min_len: int = 10,
           gap_fill: int = 5) -> list[tuple[int, int]]:
    arr = mask.astype(np.int8).copy()
    i = 0
    while i < len(arr):
        if arr[i] == 0:
            j = i
            while j < len(arr) and arr[j] == 0:
                j += 1
            if j - i <= gap_fill:
                arr[i:j] = 1
            i = max(i + 1, j)
        else:
            i += 1
    spans = []
    in_span = False
    start = 0
    for i, v in enumerate(arr):
        if v and not in_span:
            in_span, start = True, i
        elif not v and in_span:
            in_span = False
            if i - start >= min_len:
                spans.append((start, i))
    if in_span and len(arr) - start >= min_len:
        spans.append((start, len(arr)))
    return spans


def _divide_equal(x1: int, x2: int, n: int) -> list[tuple[int, int]]:
    """将 [x1, x2] 等分为 n 份。"""
    w = (x2 - x1) / n
    return [(int(x1 + i * w), int(x1 + (i + 1) * w)) for i in range(n)]


# ── 入口 ──────────────────────────────────────────────────────────────────────

def load_and_detect(img_path: Optional[Path] = None):
    if img_path is None:
        img_path = config.SCREENSHOTS_DIR / "current_screen.png"
    img = np.array(Image.open(img_path).convert("RGB"))
    det = ShelfDetector(img)
    slots = det.detect()
    return img, det, slots


def main():
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    img, det, slots = load_and_detect(img_path)

    if not slots:
        print("未检测到格子")
        return

    for s in slots:
        print(s)

    out = det.visualize(slots)
    print(f"\n可视化: {out}")


if __name__ == "__main__":
    main()
