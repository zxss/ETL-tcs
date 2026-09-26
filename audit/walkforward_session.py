"""
Walk-forward перевалидация long_overnight на ТОРГУЕМОМ окне (аудит 26.09.2026, §9 п.3а).

После перехода модели на бары основной сессии (services/session_bars.py) её цель
next_overnight = открытие основной сессии D+1 / закрытие 18:40 D — ровно то, что
держит робот. Прежние оценки силы (IC 0,051 реплея, −0,038 аудита r3) относятся к
другой цели и к рейтингу с артефактом F1; этот скрипт отвечает заново.

Протокол (зафиксирован ДО прогона):
  * тест — каждый из последних N_FOLDS календарных месяцев отдельно;
  * обучение — все окна с датой цели < начало месяца − EMBARGO_DAYS;
    масштаб признаков — только по этим окнам;
  * новая модель на каждый месяц, фиксированный seed, те же гиперпараметры,
    что в бою (hidden 32, heads 4), эпох EPOCHS;
  * сигнал — медиана предсказанной next_overnight минус поштучный круг
    (services/cost_model), как ExpPnL в бою;
  * метрики по дням: Rank IC сигнала с фактической next_overnight;
    топ-5 с порогом edge (ExpPnL > 0,5 × круг) против средней по вселенной, нетто;
  * значимость: t по дням; DSR (Bailey & López de Prado) с числом испытаний
    программы + 1;
  * ПОРОГ ДЛЯ РЕАЛЬНЫХ ДЕНЕГ (предрегистрация): t превышения топ-5 ≥ 2 И DSR ≥ 0,95
    И средний Rank IC > 0 с t ≥ 2. Иначе — только песочница.

Запуск на сервере (≈ 1 час, только чтение БД):
    cd /opt/etl-tcs && sudo -u etl .venv/bin/python -m audit.walkforward_session
Результат: audit/out/walkforward_session.json и таблица в stdout.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

N_FOLDS = int(os.getenv("WF_FOLDS", "12"))
EPOCHS = int(os.getenv("WF_EPOCHS", "20"))
EMBARGO_DAYS = 30
TOP_N = 5
EDGE_K = 0.5
N_TRIALS = int(os.getenv("WF_TRIALS", "282"))     # реестр программы 281 + это испытание
OUT = os.path.join(ROOT, "audit", "out", "walkforward_session.json")

log = logging.getLogger("wf")


def _train(X, T, Y, n_tk, epochs, seed=42):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from tft_forecast.model import TFT, quantile_loss
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TFT(X.shape[-1], n_tk, hidden=32, n_targets=Y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dl = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(T), torch.tensor(Y)),
                    batch_size=256, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, tb, yb in dl:
            opt.zero_grad()
            loss = quantile_loss(model(xb, tb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    return model


def _predict(model, X, T):
    import torch
    with torch.no_grad():
        return model(torch.tensor(X), torch.tensor(T)).cpu().numpy()


def _spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan"), len(x)
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), len(x)


def deflated_sharpe(r, n_trials):
    """DSR (Bailey & López de Prado, 2014) дневного ряда r против ожидаемого
    максимума Шарпа n_trials независимых испытаний с нулевым истинным Шарпом."""
    from scipy.stats import norm
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    t = len(r)
    if t < 10 or r.std(ddof=1) == 0:
        return float("nan"), float("nan")
    sr = r.mean() / r.std(ddof=1)
    g3 = float(((r - r.mean()) ** 3).mean() / r.std(ddof=0) ** 3)
    g4 = float(((r - r.mean()) ** 4).mean() / r.std(ddof=0) ** 4)
    var_sr = (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (t - 1)
    em = 0.5772156649
    sr0 = math.sqrt(var_sr) * ((1 - em) * norm.ppf(1 - 1 / n_trials)
                               + em * norm.ppf(1 - 1 / (n_trials * math.e)))
    return float(norm.cdf((sr - sr0) / math.sqrt(var_sr))), float(sr)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    import config
    import database
    from services import cost_model
    from tft_forecast.dataset import build_panel
    from tft_forecast.features import TARGET_COLS, load_panel
    from tft_forecast.forecast import Q_MED, T_OVN

    conn = database.get_connection()
    frames = load_panel(conn, config.TICKERS)
    panel = build_panel(frames)
    X = panel.X * panel.feat_std + panel.feat_mean            # назад к сырым признакам
    W, T, Y = panel.W, panel.T, panel.Y
    tickers = panel.tickers
    cost = np.array([cost_model.round_trip_pct(tk) for tk in tickers])
    log.info("окон %d, тикеров %d, целей %d", len(W), len(tickers), Y.shape[1])

    last = dt.date.fromordinal(int(W.max()))
    months = []
    y, m = last.year, last.month
    for _ in range(N_FOLDS):
        months.append((y, m))
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    months.reverse()

    days = []                                                   # по дням: IC, превышение
    for (y, m) in months:
        start = dt.date(y, m, 1)
        end = dt.date(y + (m == 12), m % 12 + 1, 1)
        tr = W < (start - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
        te = (W >= start.toordinal()) & (W < end.toordinal())
        if tr.sum() < 1000 or te.sum() < 50:
            log.info("%d-%02d: мало данных (обучение %d, тест %d) — пропуск",
                     y, m, tr.sum(), te.sum())
            continue
        flat = X[tr].reshape(-1, X.shape[-1])
        mu, sd = flat.mean(0), flat.std(0)
        sd[sd < 1e-8] = 1.0
        Xs = ((X - mu) / sd).astype(np.float32)
        model = _train(Xs[tr], T[tr], Y[tr].astype(np.float32), len(tickers), EPOCHS)
        pred = _predict(model, Xs[te], T[te])
        sig = pred[:, T_OVN, Q_MED] - cost[T[te]]
        real = Y[te][:, TARGET_COLS.index("next_overnight")]
        c_te = cost[T[te]]
        for w in np.unique(W[te]):
            k = W[te] == w
            if k.sum() < 10:
                continue
            s, r, c = sig[k], real[k], c_te[k]
            ic = _spearman(s, r)
            elig = np.where(s > EDGE_K * c)[0]
            top = elig[np.argsort(-s[elig])][:TOP_N] if len(elig) else np.array([], int)
            ex = float((r[top] - c[top]).mean() - (r - c).mean()) if len(top) else float("nan")
            book = float((r[top] - c[top]).mean()) if len(top) else 0.0
            days.append({"date": dt.date.fromordinal(int(w)).isoformat(), "ic": ic,
                         "excess": ex, "book_net": book, "n_top": int(len(top)),
                         "universe_net": float((r - c).mean())})
        log.info("%d-%02d: дней %d, IC %.3f", y, m,
                 sum(1 for d in days if d["date"].startswith(f"{y}-{m:02d}")),
                 np.nanmean([d["ic"] for d in days if d["date"].startswith(f"{y}-{m:02d}")]))

    ic = np.array([d["ic"] for d in days])
    ex = np.array([d["excess"] for d in days])
    book = np.array([d["book_net"] for d in days])
    t_ic, n_ic = _tstat(ic)
    t_ex, n_ex = _tstat(ex)
    dsr, sr = deflated_sharpe(book, N_TRIALS)
    passed = bool(np.isfinite(t_ex) and t_ex >= 2 and np.isfinite(dsr) and dsr >= 0.95
                  and np.nanmean(ic) > 0 and t_ic >= 2)
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
           "folds": N_FOLDS, "epochs": EPOCHS, "embargo_days": EMBARGO_DAYS,
           "days": len(days), "mean_ic": float(np.nanmean(ic)), "t_ic": t_ic,
           "ic_positive_share": float(np.nanmean(ic > 0)),
           "mean_excess_pct": float(np.nanmean(ex)), "t_excess": t_ex, "days_excess": n_ex,
           "book_net_mean_pct": float(np.nanmean(book)), "book_sharpe_daily": sr,
           "book_days_traded": int(np.sum([d["n_top"] > 0 for d in days])),
           "dsr": dsr, "n_trials": N_TRIALS, "passed_real_money_gate": passed,
           "by_day": days}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("=" * 72)
    print(f"WALK-FORWARD long_overnight, окно 18:40 → открытие основной сессии")
    print(f"месяцев {N_FOLDS}, дней {len(days)}, эмбарго {EMBARGO_DAYS} дн., эпох {EPOCHS}")
    print(f"Rank IC: среднее {res['mean_ic']:+.4f}, t {t_ic:+.2f}, "
          f"положителен в {res['ic_positive_share']:.0%} дней")
    print(f"топ-{TOP_N} − вселенная (нетто): {res['mean_excess_pct']:+.4f} %/ночь, "
          f"t {t_ex:+.2f} ({n_ex} дней)")
    print(f"книга топ-{TOP_N}: {res['book_net_mean_pct']:+.4f} %/ночь нетто, "
          f"торговых дней {res['book_days_traded']}, DSR {dsr:.3f} (испытаний {N_TRIALS})")
    print(f"ПОРОГ ДЛЯ РЕАЛЬНЫХ ДЕНЕГ: {'ПРОЙДЕН' if passed else 'НЕ ПРОЙДЕН'}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
