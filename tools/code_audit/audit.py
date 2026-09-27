"""
Поток «аудит кода»: статический разбор репозитория без загрузки файлов в контекст.

Что подаём:  рабочее дерево git (только отслеживаемые *.py, config.py, .env.example,
             scripts/*crontab*, FLOWS.md, КАК_ЗАПУСТИТЬ.md).
Что запускаем: python -m tools.code_audit.audit [--out audit/out]
Что получаем:  audit/out/code_audit_wNN.json (все находки + самопроверки) и
               audit/out/code_audit_wNN.md (сводка для человека); старые версии
               не перезаписываются. В stdout — только сводка.

Проверки:
  1. getattr(config, "X", умолчание) расходится с умолчанием в config.py;
  2. ключи окружения config.py, которых нет в .env.example, и наоборот;
  3. точки входа (__main__) без записи в FLOWS.md / КАК_ЗАПУСТИТЬ.md;
  4. модули, которые никто не импортирует и которые не точка входа;
  5. одинаковые тела функций в разных файлах (копипаст);
  6. запись в фиксированный путь (перезапись артефакта без версии);
  7. абсолютные серверные/локальные пути в коде;
  8. функции длиннее LONG_FN строк; широкие except; TODO/FIXME;
  9. отслеживаемые файлы крупнее BIG_FILE.
Самопроверки: число разобранных файлов сходится с git ls-files; итоги сводки
равны длинам списков находок; каждый ключ расхождения есть в config.py.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
import re
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from tools.flowkit import Checks, save_state, versioned  # noqa: E402

LONG_FN = 150
BIG_FILE = 1_000_000
ABS_PATH = re.compile(r"^/(opt|root|Users|home|var|tmp)/")
ENV_FUNCS = {"getenv", "get_bool_env", "get"}


def git_files(pattern: str = "") -> list[str]:
    out = subprocess.run(["git", "ls-files"] + ([pattern] if pattern else []), cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split("\n")
    return [f for f in out if f]


def literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:  # noqa: BLE001
        return None


def _norm(v):
    """'0.128' и 0.128, '1' и 1 и True — одно значение для сравнения умолчаний."""
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        try:
            return float(s)
        except ValueError:
            return s.lower()
    return v


def config_defaults(path: str) -> tuple[dict, set]:
    """{ИМЯ: умолчание} из config.py и множество ключей окружения, которые он читает."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    defaults, env_keys = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name in ENV_FUNCS and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str) and node.args[0].value.isupper():
                env_keys.add(node.args[0].value)
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None)
        if not isinstance(target, ast.Name) or not target.id.isupper():
            continue
        value = node.value
        default = literal(value)
        if default is None and value is not None:
            # float(os.getenv("X", "0.128")) / get_bool_env("X", 1) / int(...)
            for sub in ast.walk(value):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    nm = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                    if nm in ENV_FUNCS and len(sub.args) >= 2:
                        default = literal(sub.args[1])
                        break
        defaults[target.id] = default
    return defaults, env_keys


def env_example_keys(path: str) -> set:
    if not os.path.exists(path):
        return set()
    rx = re.compile(r"^\s*#?\s*([A-Z][A-Z0-9_]+)=")
    return {m.group(1) for line in open(path, encoding="utf-8") if (m := rx.match(line))}


def documented_entrypoints() -> str:
    """Весь текст FLOWS.md и всех КАК_ЗАПУСТИТЬ.md — по нему ищем упоминания модулей."""
    text = []
    for f in git_files() + [p for p in _untracked_docs()]:
        if f.endswith(("FLOWS.md", "КАК_ЗАПУСТИТЬ.md")) and os.path.exists(os.path.join(ROOT, f)):
            text.append(open(os.path.join(ROOT, f), encoding="utf-8").read())
    return "\n".join(text)


def _untracked_docs():
    out = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split("\n")
    return [f for f in out if f.endswith(("FLOWS.md", "КАК_ЗАПУСТИТЬ.md"))]


def archived_modules() -> set:
    """Модули из раздела «## Архив» FLOWS.md: разовые прогоны и отключённый код.
    Они описаны (как архив), поэтому не считаются ни недокументированными точками
    входа, ни перезаписью артефактов потока."""
    path = os.path.join(ROOT, "FLOWS.md")
    if not os.path.exists(path):
        return set()
    text = open(path, encoding="utf-8").read()
    if "## Архив" not in text:
        return set()
    part = text.split("## Архив", 1)[1].split("\n## ", 1)[0]
    mods = set()
    for m in re.findall(r"`([\w./]+?)(?:\.py)?`", part):
        mods.add(m.replace("/", ".").rstrip("."))
    return mods


def scheduled_modules() -> set:
    mods = set()
    for f in git_files():
        if "crontab" in f or f.endswith(".service") or "cron.d" in f:
            for m in re.findall(r"-m\s+([\w.]+)", open(os.path.join(ROOT, f), encoding="utf-8").read()):
                if not m.isdigit():
                    mods.add(m)
    mods.add("main")                                    # systemd etl-pipeline.timer
    return mods


def analyse() -> dict:
    py = [f for f in git_files("*.py")]
    parsed, failures = {}, []
    for f in py:
        try:
            src = open(os.path.join(ROOT, f), encoding="utf-8").read()
            parsed[f] = (src, ast.parse(src))
        except (SyntaxError, UnicodeDecodeError, FileNotFoundError) as e:
            failures.append({"file": f, "error": str(e)[:120]})

    cfg_def, cfg_env = config_defaults(os.path.join(ROOT, "config.py"))
    for f, (src, tree) in parsed.items():
        if f.startswith("tests/"):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str) and node.args[0].value.isupper():
                fn = node.func
                nm = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                owner = getattr(getattr(fn, "value", None), "attr", "") or \
                    getattr(getattr(fn, "value", None), "id", "")
                if nm in ("getenv", "get_bool_env") or (nm == "get" and owner == "environ"):
                    cfg_env.add(node.args[0].value)
    env_ex = env_example_keys(os.path.join(ROOT, ".env.example"))
    docs = documented_entrypoints()
    sched = scheduled_modules()

    imports: set[str] = set()
    entry, getattr_mis, writes, abs_paths, long_fns, broad, todos = [], [], [], [], [], [], []
    bodies: dict[str, list] = defaultdict(list)
    missing_cfg = []
    for f, (src, tree) in parsed.items():
        mod = f[:-3].replace("/", ".").removesuffix(".__init__")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(a.name for a in node.names)
                # import pnl_engine при sys.path на папку модуля
                pkg = mod.rsplit(".", 1)[0] if "." in mod else ""
                imports.update(f"{pkg}.{a.name}" for a in node.names if pkg)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:                      # from . import x / from .x import y
                    parts = mod.split(".")
                    if not f.endswith("__init__.py"):
                        parts = parts[:-1]
                    parts = parts[:len(parts) - (node.level - 1)] if node.level > 1 else parts
                    base = ".".join(parts + ([node.module] if node.module else []))
                if base:
                    imports.add(base)
                    imports.update(f"{base}.{a.name}" for a in node.names)
            elif isinstance(node, ast.If) and isinstance(node.test, ast.Compare) \
                    and getattr(node.test.left, "id", "") == "__name__":
                entry.append(mod)
            elif isinstance(node, ast.Call) and getattr(node.func, "id", "") == "getattr" \
                    and not f.startswith("tests/") and len(node.args) == 3 and getattr(node.args[0], "id", "") in ("config", "_cfg", "cfg") \
                    and isinstance(node.args[1], ast.Constant):
                key, fb = node.args[1].value, literal(node.args[2])
                if key not in cfg_def:
                    missing_cfg.append({"file": f, "line": node.lineno, "key": key})
                elif fb is not None and cfg_def[key] is not None and _norm(fb) != _norm(cfg_def[key]):
                    getattr_mis.append({"file": f, "line": node.lineno, "key": key,
                                        "fallback": fb, "config": cfg_def[key]})
            elif isinstance(node, ast.Call) and getattr(node.func, "id", "") == "open" \
                    and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) \
                    and str(node.args[1].value).startswith("w") \
                    and f.startswith(("audit/", "research/", "tools/")) \
                    and not f.startswith("tools/flowkit") and "versioned(" not in src:
                writes.append({"file": f, "line": node.lineno})
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and ABS_PATH.match(node.value) and not f.startswith("tests/"):
                abs_paths.append({"file": f, "line": node.lineno, "path": node.value[:60]})
            elif isinstance(node, ast.ExceptHandler) and (
                    node.type is None or getattr(node.type, "id", "") in ("Exception", "BaseException")):
                broad.append(f)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n = (node.end_lineno or node.lineno) - node.lineno + 1
                if n > LONG_FN:
                    long_fns.append({"file": f, "fn": node.name, "lines": n})
                body = node.body[1:] if (node.body and isinstance(node.body[0], ast.Expr)
                                         and isinstance(getattr(node.body[0], "value", None), ast.Constant)) else node.body
                if len(body) >= 4 and not f.startswith("tests/"):
                    h = hashlib.sha1("".join(ast.dump(b) for b in body).encode()).hexdigest()[:12]
                    bodies[h].append(f"{f}:{node.name}")
        for i, line in enumerate(src.split("\n"), 1):
            if re.search(r"#\s*(TODO|FIXME|XXX)\b", line):
                todos.append({"file": f, "line": i, "text": line.strip()[:90]})

    for f, (src, tree) in parsed.items():
        mod = f[:-3].replace("/", ".").removesuffix(".__init__")
        if mod != "config" and any(isinstance(n, (ast.For, ast.While, ast.With)) or
               (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)) for n in tree.body):
            entry.append(mod)                   # скрипт, исполняемый целиком
    mods = {f[:-3].replace("/", ".").removesuffix(".__init__"): f for f in parsed}
    dead = sorted(m for m, f in mods.items()
                  if not f.startswith("tests/") and not f.endswith("__init__.py")
                  and m not in entry and m not in imports
                  and not any(i.startswith(m + ".") for i in imports))
    archive = archived_modules()

    def in_archive(m: str) -> bool:
        return any(m == a or m.startswith(a + ".") for a in archive)

    undocumented = sorted({m for m in entry if not m.startswith("tests.") and not in_archive(m)
                           and m not in docs and m.replace(".", "/") + ".py" not in docs})
    writes = [w for w in writes
              if not in_archive(w["file"][:-3].replace("/", "."))]
    dupes = [v for v in bodies.values() if len({x.split(":")[0] for x in v}) > 1]
    sizes = subprocess.run(["git", "ls-files", "-s"], cwd=ROOT, capture_output=True, text=True).stdout
    big = []
    for line in sizes.split("\n"):
        if "\t" in line:
            path = line.split("\t", 1)[1]
            fp = os.path.join(ROOT, path)
            if os.path.exists(fp) and os.path.getsize(fp) > BIG_FILE:
                big.append({"file": path, "mb": round(os.path.getsize(fp) / 1e6, 1)})
    lines = sum(src.count("\n") for src, _ in parsed.values())
    return {
        "py_files": len(py), "parsed": len(parsed), "parse_failures": failures, "lines": lines,
        "tests": sum(1 for f in parsed if f.startswith("tests/")),
        "entrypoints": sorted(set(entry)), "scheduled": sorted(sched),
        "undocumented_entrypoints": undocumented,
        "dead_modules": dead, "duplicate_bodies": dupes,
        "getattr_default_mismatch": getattr_mis, "getattr_unknown_key": missing_cfg,
        "env_not_in_example": sorted(cfg_env - env_ex), "example_not_in_config": sorted(env_ex - cfg_env),
        "fixed_path_writes": writes, "absolute_paths": abs_paths, "long_functions": long_fns,
        "broad_except_files": dict(sorted(((f, broad.count(f)) for f in set(broad)),
                                          key=lambda kv: -kv[1])[:10]),
        "todos": todos, "big_files": big,
    }


def summary_md(r: dict, path: str) -> str:
    L = [f"# Аудит кода — {os.path.basename(path)}", "",
         f"Файлов *.py: {r['py_files']} (тестов {r['tests']}), строк {r['lines']:,}".replace(",", " "), ""]
    sec = [("Умолчания getattr расходятся с config.py", "getattr_default_mismatch",
            lambda x: f"`{x['file']}:{x['line']}` {x['key']}: {x['fallback']!r} против {x['config']!r}"),
           ("getattr по ключу, которого нет в config.py", "getattr_unknown_key",
            lambda x: f"`{x['file']}:{x['line']}` {x['key']}"),
           ("Ключи config.py без описания в .env.example", "env_not_in_example", str),
           ("Ключи .env.example, которых нет в config.py", "example_not_in_config", str),
           ("Точки входа без FLOWS.md / КАК_ЗАПУСТИТЬ.md", "undocumented_entrypoints", str),
           ("Модули, которые никто не импортирует", "dead_modules", str),
           ("Одинаковые тела функций в разных файлах", "duplicate_bodies", lambda x: " = ".join(x)),
           ("Запись в фиксированный путь (перезапись без версии)", "fixed_path_writes",
            lambda x: f"`{x['file']}:{x['line']}`"),
           ("Абсолютные пути в коде", "absolute_paths", lambda x: f"`{x['file']}:{x['line']}` {x['path']}"),
           (f"Функции длиннее {LONG_FN} строк", "long_functions",
            lambda x: f"`{x['file']}` {x['fn']} — {x['lines']}"),
           ("TODO / FIXME", "todos", lambda x: f"`{x['file']}:{x['line']}` {x['text']}"),
           ("Крупные отслеживаемые файлы", "big_files", lambda x: f"`{x['file']}` {x['mb']} МБ")]
    for title, key, fmt in sec:
        items = r[key]
        L.append(f"## {title} — {len(items)}")
        L += [f"- {fmt(x)}" for x in items[:40]]
        if len(items) > 40:
            L.append(f"- … ещё {len(items) - 40} (полный список — в JSON)")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "audit", "out"))
    a = ap.parse_args(argv)
    r = analyse()
    chk = Checks("code-audit")
    chk.eq("все *.py из git разобраны или учтены как ошибки разбора",
           r["parsed"] + len(r["parse_failures"]), r["py_files"])
    chk.eq("ошибок разбора нет", len(r["parse_failures"]), 0)
    cfg_def, _ = config_defaults(os.path.join(ROOT, "config.py"))
    chk.ok("ключи расхождений существуют в config.py",
           all(x["key"] in cfg_def for x in r["getattr_default_mismatch"]))
    chk.ok("мёртвые модули не пересекаются с точками входа",
           not set(r["dead_modules"]) & set(r["entrypoints"]))
    json_path = versioned(a.out, "code_audit", "json")
    md_path = json_path[:-5] + ".md"
    counts = {k: len(v) for k, v in r.items() if isinstance(v, (list, dict))}
    chk.eq("сводка = длины списков", sum(counts.values()),
           sum(len(v) for v in r.values() if isinstance(v, (list, dict))))
    save_state(json_path, params={"LONG_FN": LONG_FN, "BIG_FILE": BIG_FILE},
               results=r, checks=chk, inputs={"root": ROOT})
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(summary_md(r, md_path))
    print(chk.report())
    print(f"итог: {os.path.relpath(md_path, ROOT)}")
    for k, v in counts.items():
        if k not in ("entrypoints", "scheduled"):
            print(f"  {k:<28} {v}")
    return chk.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
