"""
auto_solver.py  ——  排序消除游戏自动通关脚本

工作流程：
  1. ADB 截图
  2. Claude Vision 解析游戏状态（容器坐标 + 物品分布）
  3. 启发式求解器选最优移动（镜像游戏内 AIBrain 的 9 策略体系）
  4. ADB swipe 执行拖拽
  5. 循环直到通关或失败

用法：
  python auto_solver.py                                # 自动启动默认包名
  python auto_solver.py --no-launch                   # 不重启游戏
  python auto_solver.py --max-steps 300 --delay 1.2   # 自定义步数/间隔
  python auto_solver.py --package com.xxx.yyy          # 指定包名
"""

import argparse
import base64
import io
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from core.device import ADBDevice
from core.controller import UIController
from core.ai_recognizer import AIRecognizer
import config


# ══════════════════════════════════════════════════════════════════════════════
# 数据模型（镜像 SimDataModel.cs）
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SlotInfo:
    item_type: Optional[str]  # None = 空槽
    is_fake: bool = False      # 混淆物（外观相同但不可消除）


@dataclass
class ContainerState:
    cid: int            # 容器序号（0-based，从左上到右下）
    x: int              # 屏幕中心 X
    y: int              # 屏幕中心 Y
    slots: list[SlotInfo] = field(default_factory=list)
    has_barrier: bool = False   # 被障碍/冰冻锁住
    has_deep_layer: bool = False  # 是否有被遮挡的下层物品

    @property
    def capacity(self) -> int:
        return len(self.slots)

    @property
    def items(self) -> list[str]:
        """顶层有效物品列表（去除混淆物）"""
        return [s.item_type for s in self.slots if s.item_type and not s.is_fake]

    @property
    def empty_count(self) -> int:
        return sum(1 for s in self.slots if s.item_type is None)

    @property
    def item_count(self) -> int:
        return sum(1 for s in self.slots if s.item_type is not None)

    def count_type(self, t: str) -> int:
        return sum(1 for s in self.slots if s.item_type == t and not s.is_fake)


@dataclass
class GameState:
    containers: list[ContainerState] = field(default_factory=list)
    level_complete: bool = False
    game_over: bool = False


@dataclass
class Move:
    src_cid: int
    dst_cid: int
    item_type: str
    reason: str
    priority: float


# ══════════════════════════════════════════════════════════════════════════════
# Vision 解析（Claude Vision → GameState）
# ══════════════════════════════════════════════════════════════════════════════

# 系统提示：告知 Claude 游戏规则和期望的 JSON 输出格式
_VISION_SYSTEM = """\
你是一款 Android 排序消除游戏（Sort & Match）的屏幕分析助手。

游戏规则：
- 屏幕上有多个"容器"（货柜/盒子），每个容器通常有 3 个槽位（也可能是 1 个）
- 玩家拖拽物品在容器之间移动
- 当同一个容器的一层中 3 个槽位全是相同物品时，自动消除
- 消除全部物品即通关

输出要求：只返回合法 JSON，不要任何其他文字：

{
  "level_complete": false,
  "game_over": false,
  "containers": [
    {
      "id": 0,
      "x": 270,
      "y": 900,
      "has_barrier": false,
      "has_deep_layer": false,
      "slots": [
        {"item": "fox",  "is_fake": false},
        {"item": "cat",  "is_fake": false},
        {"item": null,   "is_fake": false}
      ]
    }
  ]
}

字段说明：
- id：容器编号，从左上到右下，0-based
- x/y：容器在截图中的像素中心坐标（整数）
- has_barrier：容器是否被冰冻/锁链等障碍物覆盖，无法操作
- has_deep_layer：容器是否有被遮挡的下层物品（影响策略）
- slots：顶层可见槽位，item 为物品种类英文小写名称，null 表示空槽
- is_fake：混淆物（外表相同但不能消除，通常有裂纹/问号标记）

物品识别规则：
- 同种物品必须用完全相同的名称（保持一致性是关键！）
- 如果看不清，用 "unknown" 代替，不要猜测
- level_complete=true：出现通关动画/烟花/结算界面
- game_over=true：出现失败弹窗/时间耗尽/无法移动提示
"""


class GameVision:
    """用 Claude Vision 解析屏幕 → GameState"""

    def __init__(self, ai_recognizer: AIRecognizer, device: ADBDevice):
        self._ai = ai_recognizer
        self._device = device

    def analyze(self, screenshot=None) -> GameState:
        if screenshot is None:
            screenshot = self._device.screenshot()

        buf = io.BytesIO()
        Image.fromarray(screenshot).save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode()

        h, w = screenshot.shape[:2]
        # 自动检测游戏货架区域的 y 范围，用于坐标校准提示
        shelf_y_hint = self._detect_shelf_y_range(screenshot)
        user_text = (
            f"截图分辨率：宽{w}px × 高{h}px，坐标原点在左上角，x向右y向下。\n"
            f"根据颜色分析，游戏货架内容实际分布在 y={shelf_y_hint[0]}~{shelf_y_hint[1]} 范围内。\n"
            f"请确保返回的每个容器 y 坐标都在 {shelf_y_hint[0]}~{shelf_y_hint[1]} 范围内，"
            f"指向该格子内物品的中心像素位置。只返回 JSON，不要任何其他文字。"
        )
        raw = ""
        try:
            resp = self._ai._client.messages.create(
                model=self._ai._model,
                max_tokens=2048,
                system=[{"type": "text", "text": _VISION_SYSTEM,
                          "cache_control": {"type": "ephemeral"}}],
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": "image/png", "data": b64}},
                        {"type": "text", "text": user_text}
                    ]
                }]
            )
            raw = resp.content[0].text.strip()
            data = json.loads(self._extract_json(raw))
            return self._parse(data)

        except json.JSONDecodeError as e:
            logger.error(f"Vision JSON 解析失败: {e}\n原始响应(前600字):\n{raw[:600]}")
            return GameState()
        except Exception as e:
            logger.error(f"Vision 调用失败: {e}")
            return GameState()

    @staticmethod
    def _extract_json(text: str) -> str:
        """从可能包含说明文字的响应中提取 JSON 块"""
        import re
        # 优先找 ```json ... ``` 或 ``` ... ```
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if m:
            return m.group(1)
        # 找最外层 { ... }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]
        return text

    @staticmethod
    def _detect_shelf_y_range(screenshot) -> tuple[int, int]:
        """用颜色分析检测货架内容实际分布的 y 范围"""
        import numpy as np
        img = screenshot
        h = img.shape[0]
        # 背景色（米白）
        bg = np.array([222, 208, 190], dtype=float)
        results = []
        for y in range(h // 3, h * 3 // 4):
            row = img[y, 50:-50].astype(float)
            diff = np.abs(row - bg).sum(axis=1)
            non_bg = int((diff > 60).sum())
            results.append((y, non_bg))
        # 找非背景像素密集的区域
        threshold = 200
        active = [y for y, n in results if n > threshold]
        if not active:
            return (h // 3, h * 2 // 3)
        return (active[0] - 30, active[-1] + 30)

    @staticmethod
    def _parse(data: dict) -> GameState:
        state = GameState(
            level_complete=bool(data.get("level_complete", False)),
            game_over=bool(data.get("game_over", False)),
        )
        for raw_c in data.get("containers", []):
            slots = [
                SlotInfo(item_type=s.get("item"), is_fake=bool(s.get("is_fake", False)))
                for s in raw_c.get("slots", [])
            ]
            state.containers.append(ContainerState(
                cid=int(raw_c.get("id", len(state.containers))),
                x=int(raw_c.get("x", 0)),
                y=int(raw_c.get("y", 0)),
                slots=slots,
                has_barrier=bool(raw_c.get("has_barrier", False)),
                has_deep_layer=bool(raw_c.get("has_deep_layer", False)),
            ))
        logger.info(
            f"解析 {len(state.containers)} 个容器 | "
            f"通关={state.level_complete} | 失败={state.game_over}"
        )
        return state


# ══════════════════════════════════════════════════════════════════════════════
# 求解器（镜像 SimPlayer.cs 的 9 策略体系）
# ══════════════════════════════════════════════════════════════════════════════

class MoveSolver:
    """
    优先级策略（从高到低）：
      Match3    (100) — 目标有 2 同类 + 1 空位，取第 3 个
      MakeRoom  ( 90) — 目标满了有 2 同类，先搬走异类腾位
      Match2    ( 75) — 目标有 1 同类 + 空位，凑第 2 个
      Gather    ( 50) — 同类物品向数量最多的容器聚拢
      ClearDeep ( 40) — 清理深层容器的顶层（暴露隐藏物品）
      Explore   ( 10) — 从满容器移出物品增加空位
    """

    def find_move(self, state: GameState) -> Optional[Move]:
        available = [c for c in state.containers if not c.has_barrier]
        if not available:
            return None

        global_count = self._global_count(available)

        for strategy in [
            lambda: self._match3(available),
            lambda: self._make_room(available, global_count),
            lambda: self._match2(available, global_count),
            lambda: self._gather(available, global_count),
            lambda: self._clear_deep(available),
            lambda: self._explore(available),
        ]:
            moves = strategy()
            if moves:
                moves.sort(key=lambda m: -m.priority)
                return moves[0]

        return None

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _global_count(conts: list[ContainerState]) -> dict[str, int]:
        """统计所有可用容器中各物品类型的总数量"""
        cnt: dict[str, int] = defaultdict(int)
        for c in conts:
            for s in c.slots:
                if s.item_type and not s.is_fake:
                    cnt[s.item_type] += 1
        return cnt

    # ── Match3 ────────────────────────────────────────────────────────────────

    def _match3(self, conts: list[ContainerState]) -> list[Move]:
        """目标容器有 2 个同类 + 1 空位 → 找第 3 个来消除"""
        moves = []
        for tgt in conts:
            if tgt.empty_count < 1:
                continue
            for t in set(tgt.items):
                if tgt.count_type(t) != 2:
                    continue
                for src in conts:
                    if src.cid == tgt.cid:
                        continue
                    if src.count_type(t) >= 1:
                        p = 100.0
                        if src.capacity == 1:  # 单槽容器提取加分（暴露深层）
                            p += 10
                        if tgt.has_deep_layer:  # 消除后能暴露下层，链式奖励
                            p += 20
                        moves.append(Move(src.cid, tgt.cid, t,
                                          f"Match3: {t} {src.cid}→{tgt.cid}", p))
                        break  # 只需一个 src
        return moves

    # ── MakeRoom ──────────────────────────────────────────────────────────────

    def _make_room(self, conts: list[ContainerState],
                   global_count: dict[str, int]) -> list[Move]:
        """目标容器满了（有 2 同类），先搬走异类腾出空位"""
        moves = []
        for tgt in conts:
            if tgt.empty_count > 0:
                continue
            for t in set(tgt.items):
                if tgt.count_type(t) != 2:
                    continue
                # 确认其他地方有第 3 个
                third_available = any(
                    c.cid != tgt.cid and c.count_type(t) >= 1
                    for c in conts
                )
                if not third_available:
                    continue
                # 找可搬走的异类（非 t，非技能物品）
                for s in tgt.slots:
                    if not s.item_type or s.item_type == t or s.is_fake:
                        continue
                    for dst in conts:
                        if dst.cid != tgt.cid and dst.empty_count >= 1:
                            moves.append(Move(
                                tgt.cid, dst.cid, s.item_type,
                                f"MakeRoom: 挪走 {s.item_type} {tgt.cid}→{dst.cid}",
                                90.0
                            ))
                            break
                    break
        return moves

    # ── Match2 ────────────────────────────────────────────────────────────────

    def _match2(self, conts: list[ContainerState],
                global_count: dict[str, int]) -> list[Move]:
        """目标容器有 1 个同类 + 空位，聚第 2 个（全局总量 ≥ 3 才值得）"""
        moves = []
        for tgt in conts:
            if tgt.empty_count < 1:
                continue
            for t in set(tgt.items):
                if global_count.get(t, 0) < 3:
                    continue
                if tgt.count_type(t) != 1:
                    continue
                for src in conts:
                    if src.cid == tgt.cid:
                        continue
                    if src.count_type(t) >= 1:
                        p = 75.0
                        moves.append(Move(src.cid, tgt.cid, t,
                                          f"Match2: {t} {src.cid}→{tgt.cid}", p))
                        break
        return moves

    # ── Gather ────────────────────────────────────────────────────────────────

    def _gather(self, conts: list[ContainerState],
                global_count: dict[str, int]) -> list[Move]:
        """将同类物品向数量最多的容器聚拢"""
        moves = []
        # 找每种物品最多的目标容器
        item_locations: dict[str, list[int]] = defaultdict(list)
        for c in conts:
            for s in c.slots:
                if s.item_type and not s.is_fake:
                    item_locations[s.item_type].append(c.cid)

        for t, cids in item_locations.items():
            if len(cids) < 2 or global_count.get(t, 0) < 3:
                continue
            count_in = Counter(cids)
            best_cid = max(count_in, key=lambda c: count_in[c])
            tgt = next((c for c in conts if c.cid == best_cid), None)
            if not tgt or tgt.empty_count < 1:
                continue
            for src in conts:
                if src.cid == tgt.cid:
                    continue
                if src.count_type(t) >= 1:
                    p = 50.0
                    if count_in[best_cid] >= 2:
                        p += 15  # 已有 2 个，优先聚
                    moves.append(Move(src.cid, tgt.cid, t,
                                      f"Gather: {t} {src.cid}→{tgt.cid}", p))
                    break
        return moves

    # ── ClearDeep ─────────────────────────────────────────────────────────────

    def _clear_deep(self, conts: list[ContainerState]) -> list[Move]:
        """清理深层（多层）容器的顶层物品，暴露隐藏层"""
        moves = []
        deep_conts = [c for c in conts if c.has_deep_layer and c.item_count > 0]
        dst_conts = [c for c in conts if c.empty_count >= 1]
        for src in deep_conts:
            for s in src.slots:
                if not s.item_type or s.is_fake:
                    continue
                for dst in dst_conts:
                    if dst.cid != src.cid:
                        moves.append(Move(src.cid, dst.cid, s.item_type,
                                          f"ClearDeep: {s.item_type} {src.cid}→{dst.cid}",
                                          40.0))
                        break
                break
        return moves

    # ── Explore ───────────────────────────────────────────────────────────────

    def _explore(self, conts: list[ContainerState]) -> list[Move]:
        """从满容器搬出一个物品，增加可操作空间"""
        moves = []
        full = [c for c in conts if c.empty_count == 0 and c.item_count > 0]
        empty_dsts = [c for c in conts if c.empty_count >= 1]
        for src in full:
            for s in src.slots:
                if not s.item_type or s.is_fake:
                    continue
                for dst in empty_dsts:
                    if dst.cid != src.cid:
                        moves.append(Move(src.cid, dst.cid, s.item_type,
                                          f"Explore: {s.item_type} {src.cid}→{dst.cid}",
                                          10.0))
                        return moves  # 找到一个就够
        return moves


# ══════════════════════════════════════════════════════════════════════════════
# 自动玩家主控
# ══════════════════════════════════════════════════════════════════════════════

class AutoPlayer:
    def __init__(self, package_name: str = "com.tapcolor.puzzle.sort.goods.match.triple"):
        self.device = ADBDevice()
        self.ctrl = UIController(self.device)
        self.ai = AIRecognizer(self.device)
        self.vision = GameVision(self.ai, self.device)
        self.solver = MoveSolver()
        self.package = package_name
        self._stall = 0         # 连续无解计数
        self._move_log: list[tuple[int, int]] = []  # 近期 (src, dst)，用于防回弹

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        if not self.device.connect():
            logger.error("设备未连接，请运行 adb devices 检查")
            return False
        info = self.device.get_device_info()
        logger.info(f"已连接: {info['model']}  Android {info['android']}  {info['resolution']}")
        self._wake_screen()
        return True

    def _wake_screen(self):
        """唤醒屏幕并解除 MIUI 锁屏，确保 ADB 输入能到达游戏。"""
        self.device._adb("shell", "input", "keyevent", "224")   # KEYCODE_WAKEUP
        self.ctrl.wait(0.8)
        self.device._adb("shell", "wm", "dismiss-keyguard")
        self.ctrl.wait(0.5)
        logger.info("屏幕已唤醒并解锁")

    def launch(self):
        logger.info(f"重启游戏: {self.package}")
        self.device.force_stop(self.package)
        self.device.launch_app(self.package)
        self.ctrl.wait(5.0)

    def test_input(self):
        """打印触摸设备路径并做一次 raw_tap 验证注入通路是否畅通。"""
        dev = self.ctrl._get_touch_dev()
        logger.info(f"触摸设备: {dev}")
        # 点击屏幕中央（安全区域，不应触发任何游戏操作）
        ss0 = self.device.screenshot()
        h, w = ss0.shape[:2]
        cx, cy = w // 2, h // 2
        logger.info(f"测试 raw_tap 屏幕中心 ({cx},{cy})...")
        self.ctrl.raw_tap(cx, cy)
        self.ctrl.wait(0.5)
        ss1 = self.device.screenshot()
        diff = int((abs(ss0.astype(int) - ss1.astype(int))).mean())
        logger.info(f"截图均差={diff}  {'画面有变化 ✓' if diff > 3 else '画面无变化（触摸可能未生效）'}")

    def run(self, max_steps: int = 200, step_delay: float = 1.5) -> bool:
        """
        主循环，返回 True=通关，False=失败/超步。
        """
        logger.info(f"开始自动通关 | 最大步数={max_steps} | 间隔={step_delay}s")
        self.test_input()   # 验证原始触摸注入通路
        self._stall = 0
        self._move_log.clear()

        for step in range(1, max_steps + 1):
            logger.info(f"── Step {step}/{max_steps} ──────────────")

            # 1. 截图并存档
            ss = self.device.screenshot()
            self.device.save_screenshot(f"step_{step:03d}.png")

            # 2. 视觉解析
            state = self.vision.analyze(ss)

            # 3. 结束检测
            if state.level_complete:
                logger.success("通关！")
                self.device.save_screenshot("victory.png")
                return True

            if state.game_over:
                logger.warning("游戏失败")
                self.device.save_screenshot("gameover.png")
                return False

            # 4. 容器检测
            if not state.containers:
                logger.warning("未识别到容器，可能不在游戏界面，等待 2s...")
                self.ctrl.wait(2.0)
                continue

            self._log_state(state)

            # 5. 求解
            move = self.solver.find_move(state)

            if move is None:
                self._stall += 1
                logger.warning(f"无解（连续 {self._stall} 次）")
                if self._stall >= 3:
                    logger.error("连续 3 次无解，终止")
                    self.device.save_screenshot("stuck.png")
                    return False
                self.ctrl.wait(2.0)
                continue

            self._stall = 0

            # 6. 执行移动
            src = next((c for c in state.containers if c.cid == move.src_cid), None)
            dst = next((c for c in state.containers if c.cid == move.dst_cid), None)

            if src is None or dst is None:
                logger.warning(f"容器 {move.src_cid}/{move.dst_cid} 未找到，跳过")
                continue

            logger.info(f"执行 [{move.reason}]  ({src.x},{src.y}) → ({dst.x},{dst.y})")
            self._drag(src.x, src.y, dst.x, dst.y)
            self._move_log.append((move.src_cid, move.dst_cid))
            if len(self._move_log) > 10:
                self._move_log.pop(0)

            self.ctrl.wait(step_delay)

        logger.warning(f"已达最大步数 {max_steps}")
        return False

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _drag(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 800):
        """长按拖拽：先停留激活，再滑到目标"""
        self.ctrl.swipe(x1, y1, x2, y2, duration_ms)

    @staticmethod
    def _log_state(state: GameState):
        lines = []
        for c in state.containers:
            items_str = " | ".join(s.item_type or "___" for s in c.slots)
            barrier = " [锁]" if c.has_barrier else ""
            deep = " [↓]" if c.has_deep_layer else ""
            lines.append(f"  C{c.cid}({c.x},{c.y}): [{items_str}]{barrier}{deep}")
        logger.debug("当前状态:\n" + "\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="排序消除游戏自动通关")
    parser.add_argument(
        "--package", default="com.tapcolor.puzzle.sort.goods.match.triple",
        help="游戏 APK 包名"
    )
    parser.add_argument("--max-steps", type=int, default=200, help="最大操作步数")
    parser.add_argument("--delay", type=float, default=1.5, help="每步执行后的等待秒数")
    parser.add_argument("--no-launch", action="store_true", help="不自动重启游戏")
    args = parser.parse_args()

    player = AutoPlayer(args.package)

    if not player.connect():
        sys.exit(1)

    if not args.no_launch:
        player.launch()

    success = player.run(max_steps=args.max_steps, step_delay=args.delay)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
