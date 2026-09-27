"""
Исправление находки аудита «умолчание getattr расходится с config.py».

getattr(config, "X", <умолчание>) держит ВТОРОЙ источник правды: если значение в
config.py поменять, код продолжит жить со своим умолчанием в тех местах, где
config по какой-то причине без ключа (подмена в тестах, старый модуль). Самый
опасный случай из аудита 27.09.2026: getattr(config, "PROD_MAX_POSITION_RUB", 0)
в предполётной проверке — умолчание 0 выключает лимит позиции боевого счёта,
тогда как в config.py он 25 000.

Что делает: берёт последний audit/out/code_audit_wNN.json и заменяет каждое
найденное однострочное getattr(<модуль>, "X", …) на <модуль>.X. Ключ обязан
существовать в config.py (аудит это проверяет самопроверкой). Файл после правки
разбирается заново; правки, после которых код не компилируется, откатываются.

Запуск: python -m tools.code_audit.fix_getattr [--dry-run]
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def latest_audit() -> str:
    files = sorted(glob.glob(os.path.join(ROOT, "audit", "out", "code_audit_w*.json")))
    if not files:
        sys.exit("нет audit/out/code_audit_wNN.json — сначала python -m tools.code_audit.audit")
    return files[-1]


def fix_file(path: str, sites: list[dict], dry: bool) -> int:
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    lines = src.split("\n")
    want = {(s["line"], s["key"]) for s in sites}
    spans = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "getattr"
                and len(node.args) == 3 and isinstance(node.args[1], ast.Constant)
                and (node.lineno, node.args[1].value) in want
                and node.lineno == node.end_lineno):
            owner = ast.get_source_segment(src, node.args[0])
            spans.append((node.lineno, node.col_offset, node.end_col_offset,
                          f"{owner}.{node.args[1].value}"))
    # справа налево, чтобы смещения в строке не поплыли
    for ln, a, b, repl in sorted(spans, key=lambda x: (x[0], -x[1])):
        s = lines[ln - 1]
        lines[ln - 1] = s[:a] + repl + s[b:]
    new = "\n".join(lines)
    try:
        compile(new, path, "exec")
    except SyntaxError as e:
        print(f"  ✗ {os.path.relpath(path, ROOT)}: правка ломает синтаксис ({e}) — пропуск")
        return 0
    if not dry and spans:
        with open(path, "w", encoding="utf-8") as f:  # flow-state
            f.write(new)
    for ln, _a, _b, repl in spans:
        print(f"  {os.path.relpath(path, ROOT)}:{ln} → {repl}")
    return len(spans)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    audit = json.load(open(latest_audit(), encoding="utf-8"))["results"]
    by_file: dict[str, list] = {}
    for s in audit["getattr_default_mismatch"]:
        by_file.setdefault(s["file"], []).append(s)
    total = sum(fix_file(os.path.join(ROOT, f), v, a.dry_run) for f, v in sorted(by_file.items()))
    print(f"{'план' if a.dry_run else 'исправлено'}: {total} из "
          f"{len(audit['getattr_default_mismatch'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
