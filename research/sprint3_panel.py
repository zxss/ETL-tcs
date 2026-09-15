"""
Спринт 3 (ТЗ 50D): панель признаков и цели для кластерных моделей. Параметры —
research/sprint3_config.json.

Время решения — 18:30 дня t (5-минутки по бар 18:25 включительно). Вход — цена фазы
OVERNIGHT r3 (close бара 18:30), выход — та же точка через horizon торговых дней.
Цель (Hurdle Objective): y = 1, если доходность с дивидендами больше роста пая
фонда за удержание (TMON@, до 25.02.2025 LQDT) + издержки круга + запас margin;
excess = доходность − фонд − издержки — результат сделки.

Признаки строятся только по данным на момент решения:
- бумага: относительная сила к IMOEX, z к EMA, волатильность, спред Корвина–Шульца,
  ликвидность относительно своей 60-дневной истории, доля вечернего объёма вчера,
  утренний гэп и ход дня;
- рынок: IMOEX; сырьё: Brent, золото GD, юань CR — ближний контракт (выбывает за
  2 торговых дня до экспирации), доходность внутри одного контракта;
- новости: события классификатора v2 (связь «объект», лента без спама и дайджестов)
  за 48 часов до 18:30.
Сплиты: скачок open/вчерашний close вне [0,5; 2] — цены до него пересчитываются,
признаки ликвидности 60 дней после — пропуск.

Запуск (на сервере): python -m research.sprint3_panel --stage dev
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research import cost_model as cm                    # noqa: E402
from research import event_study_news as es              # noqa: E402
from research import news_classify as ncl                # noqa: E402
from research import news_event_study as ns              # noqa: E402
from research import session_timing as st                # noqa: E402
from research import sprint2_leadlag as ll               # noqa: E402

log = logging.getLogger("research.sprint3_panel")

CONFIG_PATH = os.path.join(ROOT, "research", "sprint3_config.json")
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "sprint3")
MSK = dt.timezone(dt.timedelta(hours=3))
T_DECISION = dt.time(18, 25)                 # последний бар до решения
T_ENTRY = dt.time(18, 30)                    # бар цены входа и выхода (фаза OVERNIGHT)
T_LATE = dt.time(18, 0)                      # бар цены старше — цена устарела
SPLIT_RANGE = (0.5, 2.0)
LIQ_BLACKOUT = 60
WARMUP_DAYS = 120
PRICE_COLS = ("open_main", "c1830", "c1835", "hi", "lo")
LIQ_COLS = ("amihud_rel", "adv_rel", "vol_z", "evening_share")
COMMODITIES = {"brent": "BR", "gold": "GD", "cny": "CR"}
NEWS_GROUPS = {"fin": ("FINANCIAL",), "div": ("DIVIDEND_ANNOUNCE",),
               "other": ("CORPORATE", "SANCTIONS_MACRO", "OTHER")}

DAILY_SQL = """
SELECT ticker, d,
  (array_agg(open ORDER BY ts) FILTER (WHERE t >= '10:00' AND t <= '18:25'))[1]        AS open_main,
  (array_agg(close ORDER BY ts DESC) FILTER (WHERE t >= '10:00' AND t <= '18:25'))[1]  AS c1830,
  max(t) FILTER (WHERE t >= '10:00' AND t <= '18:25')                                  AS t1830,
  (array_agg(close ORDER BY ts DESC) FILTER (WHERE t >= '10:00' AND t <= '18:30'))[1]  AS c1835,
  max(t) FILTER (WHERE t >= '10:00' AND t <= '18:30')                                  AS t1835,
  max(high) FILTER (WHERE t >= '10:00' AND t <= '18:25')                               AS hi,
  min(low) FILTER (WHERE t >= '10:00' AND t <= '18:25')                                AS lo,
  coalesce(sum(volume) FILTER (WHERE t >= '10:00' AND t <= '18:25'), 0)               AS vol_main,
  coalesce(sum(volume) FILTER (WHERE t >= '19:00'), 0)                                 AS vol_evening,
  coalesce(sum(volume), 0)                                                              AS vol_day
FROM (SELECT ticker, ts, open, high, low, close, volume,
             (ts AT TIME ZONE 'Europe/Moscow')::date AS d,
             (ts AT TIME ZONE 'Europe/Moscow')::time AS t
      FROM {table} WHERE ticker = ANY(%(tickers)s) AND ts >= %(f)s AND ts < %(t)s) s
WHERE extract(isodow FROM d) < 6
GROUP BY ticker, d
"""


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cluster_of(cfg: dict) -> dict:
    return {tk: cl for cl, tks in cfg["clusters"].items() for tk in tks}


def prepare_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["d"] = pd.to_datetime(df["d"]).dt.date
    for c in PRICE_COLS + ("vol_main", "vol_evening", "vol_day"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for col, tcol in (("c1830", "t1830"), ("c1835", "t1835")):
        fresh = df[tcol].map(lambda t: isinstance(t, dt.time) and t >= T_LATE)
        df.loc[~fresh, col] = np.nan
    return df


def load_daily(conn, table: str, tickers: list[str], d_from: dt.date, d_to: dt.date) -> pd.DataFrame:
    df = pd.read_sql(DAILY_SQL.format(table=table), conn, params={
        "tickers": list(tickers), "f": dt.datetime.combine(d_from, dt.time(), MSK),
        "t": dt.datetime.combine(d_to + dt.timedelta(days=1), dt.time(), MSK)})
    return prepare_daily(df)


# ── Бумага ───────────────────────────────────────────────────────────────────

def adjust_splits(g: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Цены до скачка вне SPLIT_RANGE переводятся в единицы после него; маска — 60 дней
    после скачка (ликвидность в лотах несравнима). Колонка adj — множитель цен строки."""
    g = g.copy()
    g["adj"] = 1.0
    ratio = g["open_main"] / g["c1835"].ffill().shift(1)
    jumps = [i for i, r in enumerate(ratio.to_numpy()) if r == r and not SPLIT_RANGE[0] <= r <= SPLIT_RANGE[1]]
    blackout = np.zeros(len(g), dtype=bool)
    for i in jumps:
        f = float(ratio.iloc[i])
        for c in PRICE_COLS + ("adj",):
            g.iloc[:i, g.columns.get_loc(c)] = g.iloc[:i][c].to_numpy() * f
        blackout[i:i + LIQ_BLACKOUT] = True
    return g, pd.Series(blackout, index=g.index)


def corwin_schultz(h: pd.Series, l: pd.Series) -> pd.Series:
    """Спред Корвина–Шульца (2012) по high/low двух соседних дней, доля цены."""
    b = np.log(h / l) ** 2 + np.log(h.shift(1) / l.shift(1)) ** 2
    gm = np.log(np.maximum(h, h.shift(1)) / np.minimum(l, l.shift(1))) ** 2
    k = 3.0 - 2.0 * np.sqrt(2.0)
    a = (np.sqrt(2.0 * b) - np.sqrt(b)) / k - np.sqrt(gm / k)
    return (2.0 * (np.exp(a) - 1.0) / (1.0 + np.exp(a))).clip(lower=0.0)


def stock_features(g: pd.DataFrame, imx: pd.Series, blackout: pd.Series | None = None) -> pd.DataFrame:
    """g — дни по календарю торгов (индекс), цены после adjust_splits; imx — IMOEX 18:30."""
    c = g["c1830"]
    lr = np.log(c / c.shift(1))
    f = pd.DataFrame(index=g.index)
    for w in (3, 10, 30):
        f[f"rs_{w}"] = ((c / c.shift(w) - 1.0) - (imx / imx.shift(w) - 1.0)) * 100.0
    f["ret_1"] = (c / c.shift(1) - 1.0) * 100.0
    sd20 = lr.rolling(20, min_periods=15).std()
    for span in (20, 50):
        ema = c.ewm(span=span, adjust=False, ignore_na=True).mean()
        f[f"z_ema{span}"] = np.log(c / ema) / sd20.replace(0, np.nan)
    f["vol_ratio"] = lr.rolling(10, min_periods=8).std() / lr.rolling(30, min_periods=20).std().replace(0, np.nan)
    pk = np.sqrt((np.log(g["hi"] / g["lo"]) ** 2).rolling(20, min_periods=15).mean() / (4.0 * np.log(2.0)))
    f["pk_over_cc"] = pk / sd20.replace(0, np.nan)
    cs = corwin_schultz(g["hi"], g["lo"])
    cs_mean = cs.rolling(20, min_periods=10).mean()
    f["cs_spread"] = cs_mean * 100.0
    f["cs_rel"] = cs / cs_mean.replace(0, np.nan)
    val = g["vol_main"] * c
    ami = lr.abs() / val.replace(0, np.nan)
    f["amihud_rel"] = np.log(ami.rolling(20, min_periods=10).median() /
                             ami.rolling(60, min_periods=40).median().replace(0, np.nan))
    f["adv_rel"] = np.log(val.rolling(5, min_periods=3).median() /
                          val.rolling(60, min_periods=40).median().replace(0, np.nan))
    vm = g["vol_main"]
    f["vol_z"] = (vm - vm.rolling(20, min_periods=15).mean()) / vm.rolling(20, min_periods=15).std().replace(0, np.nan)
    f["evening_share"] = (g["vol_evening"] / g["vol_day"].replace(0, np.nan)).shift(1)
    f["overnight_gap"] = (g["open_main"] / g["c1835"].shift(1) - 1.0) * 100.0
    f["intraday_ret"] = (c / g["open_main"] - 1.0) * 100.0
    if blackout is not None:
        f.loc[blackout.to_numpy(), list(LIQ_COLS)] = np.nan
    return f.replace([np.inf, -np.inf], np.nan)


def targets(g: pd.DataFrame, tk: str, divs: list, hurdle, cost: float, margin: float,
            horizon: int) -> pd.DataFrame:
    """Цель по цене входа close бара 18:30 (c1835) и выходу через horizon торговых дней."""
    days = list(g.index)
    c = g["c1835"].to_numpy(float)
    adj = g["adj"].to_numpy(float) if "adj" in g else np.ones(len(g))
    rows = []
    for i, d0 in enumerate(days):
        j = i + horizon
        if j >= len(days) or not (c[i] > 0) or not (c[j] > 0):
            rows.append((None, np.nan, np.nan, np.nan, np.nan))
            continue
        d1 = days[j]
        div = es.dividends_between(divs, d0, d1) * adj[i]
        r = ((c[j] + div) / c[i] - 1.0) * 100.0
        hu = float(hurdle(d0, d1))
        rows.append((d1, r, hu, float(r > hu + cost + margin), r - hu - cost))
    return pd.DataFrame(rows, index=g.index, columns=["exit_d", "R", "hurdle", "y", "excess"]).assign(cost=cost)


# ── Рынок, сырьё, новости ────────────────────────────────────────────────────

def market_features(imx: pd.Series) -> pd.DataFrame:
    lr = np.log(imx / imx.shift(1))
    return pd.DataFrame({"imoex_ret_1": (imx / imx.shift(1) - 1.0) * 100.0,
                         "imoex_ret_5": (imx / imx.shift(5) - 1.0) * 100.0,
                         "imoex_vol_20": lr.rolling(20, min_periods=15).std() * 100.0}, index=imx.index)


def commodity_features(front_bars, tdays: list[dt.date]) -> pd.DataFrame:
    """front_bars(root, d) → Bars ближнего контракта на день d; доходности внутри него."""
    sig = {"from_time": "10:00", "to_bar": T_DECISION.strftime("%H:%M")}
    rows = []
    for i, d in enumerate(tdays):
        rec = {}
        for name, root in COMMODITIES.items():
            b = front_bars(root, d)
            p = ll.close_on_day(b, d, T_DECISION) if b is not None else None
            for w in (1, 5):
                q = ll.close_on_day(b, tdays[i - w], T_DECISION) if b is not None and i >= w else None
                rec[f"{name}_ret_{w}"] = (p / q - 1.0) * 100.0 if p and q else np.nan
            s = ll.signal_2b(b, d, sig) if b is not None else None
            rec[f"{name}_day"] = s if s is not None else np.nan
        rows.append(rec)
    return pd.DataFrame(rows, index=tdays)


def news_features(events: pd.DataFrame, tickers: list[str], tdays: list[dt.date],
                  hours: int = 48) -> pd.DataFrame:
    """События за (18:30 − hours, 18:30] дня: счётчики по группам и сумма тональности."""
    at = np.array([np.datetime64(dt.datetime.combine(d, dt.time(18, 30)), "ns") for d in tdays])
    lo = at - np.timedelta64(hours, "h")
    out = []
    for tk in tickers:
        e = events[events["ticker"] == tk] if len(events) else events
        rec = {"ticker": tk, "d": tdays}
        tm_all = np.sort(pd.to_datetime(e["posted"]).to_numpy(dtype="datetime64[ns]")) if len(e) else np.array([], "datetime64[ns]")
        for name, cats in NEWS_GROUPS.items():
            tm = np.sort(pd.to_datetime(e.loc[e["category"].isin(cats), "posted"]).to_numpy(dtype="datetime64[ns]")) \
                if len(e) else np.array([], "datetime64[ns]")
            rec[f"news_{name}_48h"] = np.searchsorted(tm, at, "right") - np.searchsorted(tm, lo, "right")
        if len(e):
            srt = e.assign(_t=pd.to_datetime(e["posted"])).sort_values("_t")
            cs = np.concatenate([[0.0], np.cumsum(srt["sentiment"].to_numpy(float))])
            rec["news_sent_48h"] = cs[np.searchsorted(tm_all, at, "right")] - cs[np.searchsorted(tm_all, lo, "right")]
        else:
            rec["news_sent_48h"] = np.zeros(len(tdays))
        out.append(pd.DataFrame(rec))
    return pd.concat(out, ignore_index=True)


def market_legs(imx_entry: pd.Series, front_bars, tdays: list[dt.date], horizon: int,
                root: str) -> pd.DataFrame:
    """На день входа: доходность IMOEX и шорт-ноги (ближний фьючерс root на день входа,
    тот же контракт на выходе) за horizon торговых дней, по close бара 18:30, %."""
    rows = []
    for i, d in enumerate(tdays):
        j = i + horizon
        if j >= len(tdays):
            rows.append((np.nan, np.nan))
            continue
        a, b = imx_entry.get(d), imx_entry.get(tdays[j])
        r_idx = (b / a - 1.0) * 100.0 if a and b and a == a and b == b else np.nan
        fb = front_bars(root, d)
        p0 = ll.close_on_day(fb, d, T_ENTRY) if fb is not None else None
        p1 = ll.close_on_day(fb, tdays[j], T_ENTRY) if fb is not None else None
        rows.append((r_idx, (p1 / p0 - 1.0) * 100.0 if p0 and p1 else np.nan))
    return pd.DataFrame(rows, index=tdays, columns=["r_idx", "r_fut"])


def add_relative_targets(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """v1: цель относительно IMOEX и результаты связки (см. evaluation в конфигурации)."""
    p = panel.copy()
    m, hc, fee = cfg["target"]["margin_pct"], cfg["hedge"]["cost_rt_pct"], 2.0 * cm.FEE_SIDE_PCT
    p["alpha"] = p["R"] - p["r_idx"]
    p["y_rel"] = np.where(p["alpha"].notna(), (p["alpha"] > p["cost"] + m).astype(float), np.nan)
    p["net_alpha"] = p["alpha"] - p["cost"]
    p["net_mn"] = p["R"] - p["r_fut"] - p["hurdle"] - p["cost"] - hc
    p["net_mn_fee"] = p["R"] - p["r_fut"] - p["hurdle"] - fee - hc
    return p


# ── Сборка ───────────────────────────────────────────────────────────────────

def ticker_panel(daily_tk: pd.DataFrame, tk: str, tdays: list[dt.date], imx: pd.Series, divs: list,
                 hurdle, cost: float, tgt: dict) -> pd.DataFrame:
    g = daily_tk.set_index("d").reindex(tdays)
    g, black = adjust_splits(g)
    f = stock_features(g, imx, black)
    t = targets(g, tk, divs, hurdle, cost, tgt["margin_pct"], tgt["horizon_days"])
    return pd.concat([f, t], axis=1).assign(ticker=tk, tradable=g["c1835"].notna().to_numpy())


def build_panel(conn, cfg: dict, stage: str) -> pd.DataFrame:
    smp = cfg["samples"][stage]
    d_from, d_to = dt.date.fromisoformat(smp["from"]), dt.date.fromisoformat(smp["to"])
    lo, hi = d_from - dt.timedelta(days=WARMUP_DAYS), d_to + dt.timedelta(days=10)
    clusters = cluster_of(cfg)
    tickers = sorted(clusters)
    daily = load_daily(conn, smp["stock_table"], tickers + [es.INDEX], lo, hi)
    ix = daily[daily["ticker"] == es.INDEX].set_index("d")["c1830"]
    tdays = sorted(d for d in ix.dropna().index if d.weekday() < 5)
    imx = ix.reindex(tdays)
    imx_entry = daily[daily["ticker"] == es.INDEX].set_index("d")["c1835"].reindex(tdays)
    hedge = cfg["hedge"]["root"]
    exp, root_of = st.read_contracts()
    fut_tk = [tk for tk, r in root_of.items() if r in set(COMMODITIES.values()) | {hedge}]
    fd = st.load_daily(conn, "research_fut_5m", fut_tk, lo - dt.timedelta(days=10), hi)
    fd["root"] = fd["ticker"].map(root_of)
    front = st.front_contracts(fd, ll.eligible_until(exp, 2))
    fmap = {(r.root, r.d): r.front for r in front.itertuples()}
    fbars = es.load_bars(conn, "research_fut_5m", sorted(set(front["front"])), lo, hi)
    posts = ns.load_posts(conn, "markettwits", lo - dt.timedelta(days=3))
    posts = posts[posts["msk"].dt.date <= d_to]
    events, cnt = es.build_events(posts, ncl.Classifier())
    log.info("постов %d, событий %d", len(posts), len(events))
    divs = es.load_dividends(es.DIV_PATH, tdays)
    hurdle = es.Hurdle(es.HURDLE_PATH)
    spreads = cm.load_spreads()
    tgt = cfg["target"]
    parts = []
    for tk in tickers:
        dtk = daily[daily["ticker"] == tk]
        if dtk.empty:
            continue
        cost = cm.round_trip(tk, tgt["cost_scenario"], spreads)
        parts.append(ticker_panel(dtk, tk, tdays, imx, divs.get(tk, []), hurdle.growth, cost, tgt)
                     .rename_axis("d").reset_index())
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.merge(market_features(imx).rename_axis("d").reset_index(), on="d", how="left")
    panel = panel.merge(commodity_features(lambda r, d: fbars.get(fmap.get((r, d))), tdays)
                        .rename_axis("d").reset_index(), on="d", how="left")
    panel = panel.merge(news_features(events, tickers, tdays), on=["ticker", "d"], how="left")
    legs = market_legs(imx_entry, lambda r, d: fbars.get(fmap.get((r, d))), tdays, tgt["horizon_days"], hedge)
    panel = add_relative_targets(panel.merge(legs.rename_axis("d").reset_index(), on="d", how="left"), cfg)
    panel["cluster"] = panel["ticker"].map(clusters)
    panel = panel[(panel["d"] >= d_from) & (panel["d"] <= d_to) & panel["tradable"]]
    return panel.reset_index(drop=True)


def feature_list(cfg: dict) -> list[str]:
    f = cfg["features"]
    return f["stock"] + f["market"] + f["commodity"] + f["news"]


def panel_path(stage: str, version: str) -> str:
    return os.path.join(OUT_DIR, f"panel_{stage}_{version}.csv.gz")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Спринт 3: панель признаков и цели")
    ap.add_argument("--stage", choices=("dev", "holdout"), required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    if a.stage == "holdout" and not cfg["samples"]["holdout"].get("approved"):
        raise SystemExit("протокол отложенной выборки не утверждён пользователем")
    import database
    conn = database.get_connection()
    try:
        panel = build_panel(conn, cfg, a.stage)
    finally:
        conn.close()
    os.makedirs(OUT_DIR, exist_ok=True)
    panel.to_csv(panel_path(a.stage, cfg["version"]), index=False, float_format="%.6g")
    y = cfg["target"].get("column", "y")
    log.info("панель %s: строк %d, бумаг %d, дат %d, доля %s=1 %.3f", a.stage, len(panel),
             panel["ticker"].nunique(), panel["d"].nunique(), y, panel[y].mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
