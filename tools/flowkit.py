"""
Общие детали «потоков» (FLOWS.md): версии артефактов и встроенная самопроверка.

Правила проекта (CLAUDE.md):
  * артефакт не перезаписывается — каждая версия ложится рядом как <имя>_wNN.<ext>;
  * параметры и итоги прогона сохраняются в JSON рядом с артефактом;
  * скрипт сам пересчитывает ключевые итоги и пишет проверки; упавшая проверка —
    ненулевой код выхода, а не повод перепроверять глазами.

Пример:
    from tools.flowkit import Checks, versioned, save_state
    chk = Checks("vrp-backtest")
    chk.eq("циклов RI = 153", n_ri, 153)
    chk.close("сумма по годам = итог", sum(by_year), total, tol=1e-9)
    out = versioned("audit/out", "vrp_backtest", "json")      # …/vrp_backtest_w03.json
    save_state(out, params=PARAMS, results=res, checks=chk)
    raise SystemExit(chk.exit_code())
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import subprocess


def versioned(directory: str, stem: str, ext: str) -> str:
    """Следующий свободный путь <directory>/<stem>_wNN.<ext> (NN от 01)."""
    os.makedirs(directory, exist_ok=True)
    rx = re.compile(rf"^{re.escape(stem)}_w(\d+)\.{re.escape(ext)}$")
    nums = [int(m.group(1)) for f in os.listdir(directory) if (m := rx.match(f))]
    return os.path.join(directory, f"{stem}_w{(max(nums) + 1 if nums else 1):02d}.{ext}")


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "?"
    except Exception:  # noqa: BLE001 — на сервере без .git
        return "?"


class Checks:
    """Журнал самопроверок: каждая проверка — имя, результат, детали."""

    def __init__(self, flow: str):
        self.flow = flow
        self.items: list[dict] = []

    def ok(self, name: str, cond: bool, detail: str = "") -> bool:
        self.items.append({"name": name, "ok": bool(cond), "detail": str(detail)})
        return bool(cond)

    def eq(self, name: str, got, want) -> bool:
        return self.ok(name, got == want, f"получено {got!r}, ожидалось {want!r}")

    def close(self, name: str, got: float, want: float, tol: float = 1e-6) -> bool:
        good = (got is not None and want is not None
                and math.isfinite(got) and math.isfinite(want) and abs(got - want) <= tol)
        return self.ok(name, good, f"{got} против {want} (допуск {tol})")

    @property
    def failed(self) -> list[dict]:
        return [c for c in self.items if not c["ok"]]

    def report(self) -> str:
        head = f"[{self.flow}] проверок {len(self.items)}, провалено {len(self.failed)}"
        lines = [head] + [f"  ✗ {c['name']}: {c['detail']}" for c in self.failed]
        return "\n".join(lines)

    def exit_code(self) -> int:
        return 1 if self.failed else 0

    def as_dict(self) -> dict:
        return {"flow": self.flow, "total": len(self.items),
                "failed": len(self.failed), "items": self.items}


def save_state(path: str, *, params: dict, results, checks: Checks | None = None,
               inputs: dict | None = None) -> str:
    """Параметры + итоги + проверки + происхождение — одним JSON рядом с артефактом."""
    doc = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
           "git": git_rev(), "inputs": inputs or {}, "params": params,
           "checks": checks.as_dict() if checks else None, "results": results}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1, default=str)
    return path
