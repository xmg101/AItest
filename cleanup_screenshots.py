"""
清理 screenshots/ 目录中未被任何 .py 文件引用的截图。

用法:
  python cleanup_screenshots.py           # 直接删除
  python cleanup_screenshots.py --dry-run # 预览，不实际删除
"""

import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
SCREENSHOTS_DIR = BASE_DIR / "screenshots"


def collect_referenced_names(py_files: list[Path]) -> set[str]:
    pattern = re.compile(r'[\w\-\.]+\.png')
    refs: set[str] = set()
    for f in py_files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        refs.update(pattern.findall(text))
    return refs


def main():
    dry_run = "--dry-run" in sys.argv

    if not SCREENSHOTS_DIR.exists():
        print("screenshots/ 目录不存在，无需清理。")
        return

    py_files = list(BASE_DIR.rglob("*.py"))
    referenced = collect_referenced_names(py_files)

    pngs = sorted(SCREENSHOTS_DIR.glob("*.png"))
    if not pngs:
        print("screenshots/ 目录为空，无需清理。")
        return

    to_delete = [p for p in pngs if p.name not in referenced]
    kept = len(pngs) - len(to_delete)

    if not to_delete:
        print(f"所有 {len(pngs)} 张截图均被代码引用，无需清理。")
        return

    mode = "（演习模式，不实际删除）" if dry_run else ""
    print(f"共 {len(pngs)} 张截图，{kept} 张被引用，{len(to_delete)} 张未引用{mode}：\n")
    for p in to_delete:
        print(f"  {'[预览]' if dry_run else '[删除]'} {p.name}")
        if not dry_run:
            p.unlink()

    if not dry_run:
        print(f"\n已删除 {len(to_delete)} 张未引用截图，保留 {kept} 张。")
    else:
        print(f"\n演习完成。去掉 --dry-run 参数可实际执行删除。")


if __name__ == "__main__":
    main()
