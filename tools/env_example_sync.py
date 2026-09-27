"""
Поток «синхронизация .env.example»: ключи окружения, которые читает код, но
которых нет в .env.example, дописываются АВТОМАТИЧЕСКИ одним блоком.

Что подаём:  config.py (умолчания) + код (кто ещё читает окружение) + .env.example.
Что запускаем: python -m tools.env_example_sync [--check]
Что получаем:  блок между маркерами «>>> авто» / «<<< авто» в конце .env.example —
               пересобирается целиком при каждом запуске (идемпотентно).
               --check: только проверить, код выхода 1, если блок устарел.

Строки блока ЗАКОММЕНТИРОВАНЫ («# КЛЮЧ=умолчание»): это документация, а не
настройка. Хук выкладки сверяет только незакомментированные ключи, поэтому блок
не порождает предупреждений «ENV-ДРЕЙФ», а аудит кода (tools/code_audit) считает
ключ описанным. Умолчание берётся из config.py — там единственный источник правды.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.code_audit.audit import config_defaults, env_example_keys, git_files  # noqa: E402

BEGIN = "# >>> авто: ключи, которые читает код, но не описаны выше (python -m tools.env_example_sync)"
END = "# <<< авто"
ENV_READ = re.compile(r"""(?:getenv|get_bool_env|environ\.get)\(\s*["']([A-Z][A-Z0-9_]+)["']""")


def code_env_keys() -> dict[str, str]:
    """{КЛЮЧ: первый файл, где его читают} по всему отслеживаемому коду (без тестов)."""
    out: dict[str, str] = {}
    for f in git_files("*.py"):
        if f.startswith("tests/"):
            continue
        for k in ENV_READ.findall(open(os.path.join(ROOT, f), encoding="utf-8").read()):
            out.setdefault(k, f)
    return out


def build_block(example_path: str) -> tuple[str, list[str]]:
    text = open(example_path, encoding="utf-8").read()
    manual = text.split(BEGIN)[0]
    documented = env_example_keys_from_text(manual)
    defaults, _ = config_defaults(os.path.join(ROOT, "config.py"))
    keys = code_env_keys()
    missing = sorted(k for k in keys if k not in documented)
    lines = [BEGIN,
             "# Сгенерировано из config.py: значение — умолчание кода. Раскомментируйте,",
             "# чтобы переопределить. Руками этот блок не править — он пересобирается."]
    for k in missing:
        d = defaults.get(k)
        if isinstance(d, list):
            d = " ".join(map(str, d))
        val = "" if d is None else d
        where = "config.py" if k in defaults else keys[k]
        lines.append(f"# {k}={val}".ljust(46) + f"  # {where}")
    lines.append(END)
    return manual.rstrip("\n") + "\n\n" + "\n".join(lines) + "\n", missing


def env_example_keys_from_text(text: str) -> set:
    rx = re.compile(r"^\s*#?\s*([A-Z][A-Z0-9_]+)=")
    return {m.group(1) for line in text.split("\n") if (m := rx.match(line))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    path = os.path.join(ROOT, ".env.example")
    new, missing = build_block(path)
    old = open(path, encoding="utf-8").read()
    if a.check:
        stale = new != old
        print(f".env.example: {'УСТАРЕЛ' if stale else 'актуален'} (ключей в авто-блоке {len(missing)})")
        return 1 if stale else 0
    if new != old:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new)
    # самопроверка: после записи все ключи кода описаны
    left = set(code_env_keys()) - env_example_keys(path)
    print(f"авто-блок: {len(missing)} ключей; неописанных после записи: {len(left)}")
    return 1 if left else 0


if __name__ == "__main__":
    raise SystemExit(main())
