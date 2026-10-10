"""
Загрузка дневных свечей Binance spot (пары к USDT) из архива data.binance.vision —
включая делистингованные пары (список берётся из листинга S3-бакета).

Что подаём:  ничего (публичный архив, ключей не нужно).
Что запускаем: python -m research.crypto.download --out <каталог> [--workers 8] [--limit N]
Что получаем:  <out>/raw/<SYMBOL>/<SYMBOL>-1d-YYYY-MM.zip (кэш, повторно не качается),
               <out>/daily/<SYMBOL>.csv — date,open,high,low,close,quote_volume
               <out>/symbols.json — список пар и исключённых с причиной.
С 2025 года время в архиве в микросекундах, раньше — в миллисекундах.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DL = "https://data.binance.vision/"
API = "https://data-api.binance.vision/api/v3/klines"   # отдаёт и делистингованные пары
PREFIX = "data/spot/monthly/klines/"
LAST_MONTH = "2026-09"

STABLE = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "PAX", "UST", "USTC", "EUR", "AEUR", "GBP",
          "TRY", "BRL", "AUD", "RUB", "UAH", "NGN", "BIDR", "IDRT", "BVND", "ZAR", "PLN", "RON", "ARS",
          "JPY", "MXN", "COP", "CZK", "SUSD", "USDS", "USDSB", "PAXG", "XUSD", "USD1", "EURI", "USDE",
          "BFUSD", "RLUSD", "PYUSD", "USDF", "FRAX", "LUSD", "GUSD"}
WRAPPED = {"WBTC", "WBETH", "BETH", "WETH", "BBTC", "STETH", "WSTETH", "BNSOL"}
LEVERAGED = re.compile(r"(UP|DOWN|BULL|BEAR)$")


def get(url: str, tries: int = 5) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "etl-tcs-research"}),
                                        timeout=60) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def list_prefix(prefix: str, delimiter: bool = True) -> tuple[list[str], list[str]]:
    """Все CommonPrefixes и Keys под префиксом (с пагинацией по marker)."""
    dirs, keys, marker = [], [], ""
    while True:
        q = {"prefix": prefix, "marker": marker}
        if delimiter:
            q["delimiter"] = "/"
        x = get(f"{S3}?{urllib.parse.urlencode(q)}").decode()
        dirs += re.findall(r"<Prefix>([^<]+/)</Prefix>", x)[1:] if delimiter else []
        keys += re.findall(r"<Key>([^<]+)</Key>", x)
        if "<IsTruncated>true</IsTruncated>" not in x:
            break
        nm = re.search(r"<NextMarker>([^<]+)</NextMarker>", x)
        marker = nm.group(1) if nm else (keys[-1] if keys else dirs[-1])
    return [d for d in dirs if d != prefix], keys


def base_of(sym: str) -> str | None:
    return sym[:-4] if sym.endswith("USDT") else None


def exclusion(sym: str) -> str | None:
    b = base_of(sym)
    if b is None:
        return "не USDT"
    if b in STABLE:
        return "стейблкоин/фиат"
    if b in WRAPPED:
        return "обёрнутый"
    if LEVERAGED.search(b) and b not in ("JUP",):
        return "плечевой токен"
    return None


def to_ts(x: str) -> dt.date:
    v = int(x)
    v = v / 1_000_000 if v > 10 ** 14 else v / 1000          # микросекунды с 2025 г.
    return dt.datetime.fromtimestamp(v, tz=dt.timezone.utc).date()


def fetch_api(sym: str) -> dict:
    """Дневные свечи через API порциями по 1000 дней (тот же источник, что архив)."""
    rows, start = {}, int(dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
    end_ms = int(dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
    while start < end_ms:
        q = urllib.parse.urlencode({"symbol": sym, "interval": "1d", "startTime": start, "endTime": end_ms - 1,
                                    "limit": 1000})
        data = json.loads(get(f"{API}?{q}"))
        if not isinstance(data, list) or not data:
            break
        for r in data:
            rows[to_ts(str(r[0]))] = (r[1], r[2], r[3], r[4], r[7])
        start = int(data[-1][0]) + 86_400_000
        if len(data) < 1000:
            break
    return rows


def fetch_symbol(sym: str, out: str) -> tuple[str, int]:
    try:
        rows = fetch_api(sym)
    except Exception:  # noqa: BLE001
        rows = {}
    if rows:
        return write_daily(sym, out, rows)
    return fetch_zip(sym, out)


def write_daily(sym: str, out: str, rows: dict) -> tuple[str, int]:
    dest = os.path.join(out, "daily", f"{sym}.csv")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", newline="") as f:  # flow-state: дневной ряд пары
        w = csv.writer(f)
        w.writerow(["date", "open", "high", "low", "close", "quote_volume"])
        for d in sorted(rows):
            if d <= dt.date(2026, 9, 30):
                w.writerow([d.isoformat(), *rows[d]])
    return sym, len(rows)


def fetch_zip(sym: str, out: str) -> tuple[str, int]:
    _, keys = list_prefix(f"{PREFIX}{sym}/1d/", delimiter=False)
    zips = sorted(k for k in keys if k.endswith(".zip") and k[-11:-4] <= LAST_MONTH)
    rows = {}
    rawdir = os.path.join(out, "raw", sym)
    os.makedirs(rawdir, exist_ok=True)
    for k in zips:
        path = os.path.join(rawdir, os.path.basename(k))
        if not os.path.exists(path):
            data = get(DL + k)
            with open(path + ".tmp", "wb") as f:  # flow-state: кэш архива
                f.write(data)
            os.replace(path + ".tmp", path)
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                for r in csv.reader(io.TextIOWrapper(z.open(name), encoding="utf-8")):
                    if not r or not r[0].isdigit():
                        continue
                    rows[to_ts(r[0])] = (r[1], r[2], r[3], r[4], r[7])
    return write_daily(sym, out, rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="только первые N пар (проверка)")
    ap.add_argument("--only", default="", help="пары через запятую")
    a = ap.parse_args(argv)
    dirs, _ = list_prefix(PREFIX)
    syms = sorted(d[len(PREFIX):-1] for d in dirs)
    usdt = [s for s in syms if s.endswith("USDT")]
    excl = {s: exclusion(s) for s in usdt if exclusion(s)}
    keep = [s for s in usdt if s not in excl]
    if a.only:
        keep = [s for s in a.only.split(",") if s in usdt]
    if a.limit:
        keep = keep[: a.limit]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "symbols.json"), "w") as f:  # flow-state
        json.dump({"listed_at": dt.datetime.now(dt.timezone.utc).isoformat(), "all_usdt": len(usdt),
                   "keep": keep, "excluded": excl}, f, ensure_ascii=False, indent=1)
    print(f"пар к USDT в архиве {len(usdt)}, исключено {len(excl)}, качаем {len(keep)}", flush=True)
    done, bad = 0, []
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch_symbol, s, a.out): s for s in keep}
        for fu in cf.as_completed(futs):
            try:
                fu.result()
            except Exception as e:  # noqa: BLE001
                bad.append((futs[fu], str(e)[:120]))
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(keep)}", flush=True)
    print(f"готово: {done - len(bad)} пар, ошибок {len(bad)}", bad[:5])
    return 1 if len(bad) > len(keep) * 0.02 else 0


if __name__ == "__main__":
    raise SystemExit(main())
