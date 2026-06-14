"""
feature_engineering.py — расширенный Feature Engineering для TFT (MOEX).

Добавляет 4 группы признаков к дневному датафрейму свечей и формирует
конфигурацию pytorch-forecasting.TimeSeriesDataSet с КОРРЕКТНЫМ распределением
признаков по типам (known/unknown, real/categorical).

Блоки признаков
---------------
  1) Календарные / временные  — KNOWN FUTURE (известны заранее):
       day_of_week, is_weekend_ahead, days_to_moex_expiration
  2) Микроструктура / ликвидность — PAST OBSERVED (unknown):
       amihud_illiquidity, volume_z_score, vwap_distance
  3) Макро-контекст РФ — PAST OBSERVED (unknown), мерж по дате:
       imoex_return, rgbi_yield_change, rvi_return
  4) Стационарность — PAST OBSERVED (unknown):
       log_return (основной ценовой вход / возможный target),
       atr_percentile_252

Публичный API
-------------
  build_tft_features(df, macro_df) -> pd.DataFrame
  make_timeseries_dataset(df, ...) -> TimeSeriesDataSet  (нужен pytorch-forecasting)
  FEATURE_SPEC  — словарь с распределением признаков по типам (для прозрачности).

ВАЖНО про отсутствие look-ahead:
  Все past-observed признаки считаются ТОЛЬКО по прошлым/текущим данным
  (rolling/shift(1)). Календарные known-признаки детерминированы из даты, их
  «знание будущего» легитимно (день недели и дата экспирации известны заранее).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

# ── Конфигурация мерджа макро-данных ────────────────────────────────────────────
# Имена индексов, которые ищем в macro_df (регистронезависимо, по подстроке).
_MACRO_INDEXES = ("IMOEX", "RGBI", "RVI")

OHLCV = ["open", "high", "low", "close", "volume"]


# ════════════════════════════════════════════════════════════════════════════════
#  Блок 1 — Календарь и экспирации MOEX (KNOWN FUTURE)
# ════════════════════════════════════════════════════════════════════════════════

def third_thursday(year: int, month: int) -> dt.date:
    """
    Дата 3-го четверга месяца — день экспирации квартальных фьючерсов MOEX.

    Логика: четверг имеет индекс weekday()==3 (Пн=0 … Вс=6). Берём 1-е число
    месяца, считаем смещение до ПЕРВОГО четверга: (3 - weekday(1-е)) % 7.
    Третий четверг = первый четверг + 14 дней. Всегда попадает в диапазон 15..21.
    """
    first = dt.date(year, month, 1)
    offset = (3 - first.weekday()) % 7      # дней от 1-го числа до первого четверга
    first_thursday_day = 1 + offset
    return dt.date(year, month, first_thursday_day + 14)


def _days_to_next_expiration(d: dt.date) -> int:
    """
    Календарных дней до ближайшей квартальной экспирации MOEX (3-й четверг
    марта/июня/сентября/декабря).

    Перебираем экспирации текущего и СЛЕДУЮЩЕГО года (чтобы корректно
    «перепрыгнуть» декабрь → март), оставляем те, что не раньше d, берём
    минимальную. В день экспирации результат = 0.
    """
    candidates = [
        third_thursday(year, month)
        for year in (d.year, d.year + 1)
        for month in (3, 6, 9, 12)
    ]
    future = [exp for exp in candidates if exp >= d]
    nearest = min(future)
    return (nearest - d).days


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    dow = df["date"].dt.dayofweek                      # 0=Пн … 4=Пт (6=Вс на случай данных выходных)
    df["day_of_week"] = dow.astype(int)
    # Пятница → переносим позицию через выходные → премия/риск weekend.
    df["is_weekend_ahead"] = (dow == 4).astype(int)

    # days_to_moex_expiration считаем по УНИКАЛЬНЫМ датам (их немного) и мапим
    # на все строки — это в разы быстрее, чем apply по каждой строке тикера.
    uniq = pd.Index(df["date"].dt.normalize().unique())
    mapping = {ts: _days_to_next_expiration(pd.Timestamp(ts).date()) for ts in uniq}
    df["days_to_moex_expiration"] = df["date"].dt.normalize().map(mapping).astype(int)
    return df


# ════════════════════════════════════════════════════════════════════════════════
#  Блок 2 — Микроструктура и ликвидность (PAST OBSERVED, по каждому тикеру)
# ════════════════════════════════════════════════════════════════════════════════

def _add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Считается ПО КАЖДОМУ тикеру (groupby-transform), чтобы окна и сдвиги
    не «протекали» между инструментами. df отсортирован по ['ticker','date']."""
    gb = df.groupby("ticker", sort=False)
    log_ret = np.log(df["close"] / gb["close"].shift(1))

    # Неликвидность Амихуда: |доходность| на единицу объёма. Микро-константа
    # 1e-8 в знаменателе защищает от деления на ноль в дни с нулевым объёмом.
    # Масштаб 1e9: у голубых фишек MOEX объёмы огромны, поэтому «сырой» Амихуд
    # ~1e-10 и для сети неотличим от нуля — домножаем на 1e9 (классическая
    # нормировка Amihud 2002), чтобы признак жил в читаемом диапазоне.
    # log1p поверх масштаба: распределение Амихуда сильно скошено вправо (дни
    # near-нулевого объёма дают всплески ×10^4) — лог сжимает хвост в почти
    # лог-нормальный вид, устойчивый к выбросам. log1p(x)=ln(1+x), при x≥0
    # монотонен и log1p(0)=0, поэтому нули остаются нулями.
    AMIHUD_SCALE = 1e9
    amihud_raw = log_ret.abs() / (df["volume"] + 1e-8) * AMIHUD_SCALE
    df["amihud_illiquidity"] = np.log1p(amihud_raw)

    # Z-оценка объёма: насколько сегодняшний объём отклонился от нормы за 20 дней.
    vol_sma = gb["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    vol_std = gb["volume"].transform(lambda s: s.rolling(20, min_periods=5).std())
    df["volume_z_score"] = (df["volume"] - vol_sma) / vol_std.replace(0, np.nan)

    # VWAP-дистанция: отклонение close от типичной цены (H+L+C)/3 — дешёвая
    # аппроксимация дневного VWAP при отсутствии тиковых данных (row-wise).
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    df["vwap_distance"] = (df["close"] - typical_price) / typical_price
    return df


# ════════════════════════════════════════════════════════════════════════════════
#  Блок 4 — Стационарность (PAST OBSERVED, по каждому тикеру)
#  (определён до Блока 3, т.к. логически тоже на уровне тикера)
# ════════════════════════════════════════════════════════════════════════════════

def _rolling_percentile(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """
    Перцентиль (0..1) ПОСЛЕДНЕГО значения каждого окна относительно самого окна.

    pandas>=1.4: Rolling.rank(pct=True) возвращает ранг последнего элемента окна
    в долях — это ровно «в каком процентиле сидит текущий ATR за последние N дней».
    Фолбэк через scipy.percentileofscore — на случай старого pandas (медленнее).
    """
    try:
        return s.rolling(window, min_periods=min_periods).rank(pct=True)
    except (AttributeError, ValueError):
        from scipy.stats import percentileofscore
        return s.rolling(window, min_periods=min_periods).apply(
            lambda w: percentileofscore(w, w[-1], kind="mean") / 100.0, raw=True
        )


def _add_stationarity_features(df: pd.DataFrame) -> pd.DataFrame:
    """По каждому тикеру (groupby-transform). df отсортирован по ['ticker','date']."""
    gb = df.groupby("ticker", sort=False)
    prev_close = gb["close"].shift(1)

    # Основной стационарный ценовой вход (вместо «сырого» рублёвого close).
    df["log_return"] = np.log(df["close"] / prev_close)

    # ATR(14) по истинному диапазону (True Range), prev_close — в разрезе тикера.
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.groupby(df["ticker"], sort=False).transform(
        lambda s: s.rolling(14, min_periods=5).mean()
    )

    # Где текущий ATR относительно последних 252 торговых дней (0..1).
    df["atr_percentile_252"] = atr14.groupby(df["ticker"], sort=False).transform(
        lambda s: _rolling_percentile(s, window=252, min_periods=20)
    )
    return df


# ════════════════════════════════════════════════════════════════════════════════
#  Блок 3 — Макро-контекст РФ (PAST OBSERVED, общий по рынку, мерж по дате)
# ════════════════════════════════════════════════════════════════════════════════

def _find_macro_col(macro_df: pd.DataFrame, index_name: str) -> str | None:
    """Ищет колонку индекса по подстроке (регистронезависимо). Предпочитает
    колонку, содержащую 'close'; иначе первую подходящую."""
    name = index_name.lower()
    cands = [c for c in macro_df.columns if name in str(c).lower()]
    if not cands:
        return None
    close_first = [c for c in cands if "close" in str(c).lower()]
    return (close_first or cands)[0]


def _build_macro_frame(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Считает макро-доходности ОДИН раз на общем календаре (индексы единые для
    всего рынка), затем эти значения мерджатся в каждый тикер по дате.
    """
    m = macro_df.copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values("date").reset_index(drop=True)

    imoex = _find_macro_col(m, "IMOEX")
    rgbi = _find_macro_col(m, "RGBI")
    rvi = _find_macro_col(m, "RVI")

    out = pd.DataFrame({"date": m["date"]})
    # Лог-доходность индекса Мосбиржи — общий риск-он/риск-офф рынка.
    out["imoex_return"] = (
        np.log(m[imoex] / m[imoex].shift(1)) if imoex else np.nan
    )
    # Изменение RGBI (индекс гособлигаций): прокси жёсткости ДКП ЦБ РФ.
    # Берём простую разницу уровней, как в ТЗ (это индекс цен, не доходностей).
    out["rgbi_yield_change"] = (
        m[rgbi] - m[rgbi].shift(1) if rgbi else np.nan
    )
    # Лог-доходность RVI — «индекс страха», скачки = паника в системе.
    out["rvi_return"] = (
        np.log(m[rvi] / m[rvi].shift(1)) if rvi else np.nan
    )
    return out


def _merge_macro(df: pd.DataFrame, macro_df: pd.DataFrame | None) -> pd.DataFrame:
    macro_cols = ["imoex_return", "rgbi_yield_change", "rvi_return"]
    if macro_df is None or len(macro_df) == 0:
        for c in macro_cols:
            df[c] = 0.0          # макро недоступно — нейтральные нули
        return df
    macro = _build_macro_frame(macro_df)
    df = df.merge(macro, on="date", how="left")
    return df


# ════════════════════════════════════════════════════════════════════════════════
#  Оркестратор
# ════════════════════════════════════════════════════════════════════════════════

# Признаки, для которых leading-NaN осмысленно заполнять нулём (это доходности /
# отклонения — «нейтраль» = 0). Уровневые признаки (перцентиль) — ffill затем 0.
_ZERO_FILL = [
    "log_return", "amihud_illiquidity", "volume_z_score", "vwap_distance",
    "imoex_return", "rgbi_yield_change", "rvi_return",
]
_FFILL_THEN_ZERO = ["atr_percentile_252"]


def build_tft_features(df: pd.DataFrame, macro_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Главная функция. Принимает:
      df       — колонки: date, ticker, open, high, low, close, volume
      macro_df — колонки: date + котировки индексов IMOEX / RGBI / RVI
                 (имена ищутся по подстроке; close-колонка предпочтительна)

    Возвращает df с добавленными признаками 4 блоков, заполненными NaN и
    колонкой time_idx (целочисленная ось времени на тикер, нужна TFT).
    """
    if df.empty:
        raise ValueError("build_tft_features: пустой df")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # — Блок 1: календарь (детерминирован из даты, без группировки) —
    df = _add_calendar_features(df)

    # — Блоки 2 и 4: микроструктура и стационарность ПО КАЖДОМУ тикеру —
    # через groupby-transform (без .apply на грппирующих колонках —
    # совместимо со всеми версиями pandas, без FutureWarning).
    df = _add_microstructure_features(df)
    df = _add_stationarity_features(df)

    # — Блок 3: макро по дате (общий календарь) —
    df = _merge_macro(df, macro_df)

    # — time_idx: целочисленная ось времени внутри каждого тикера (0,1,2,…) —
    # TimeSeriesDataSet требует МОНОТОННЫЙ целочисленный time_idx на группу.
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["time_idx"] = df.groupby("ticker").cumcount().astype(int)

    # — Обработка NaN —
    # Доходности/отклонения: leading-NaN → 0 (нейтрально).
    for c in _ZERO_FILL:
        if c in df.columns:
            df[c] = df.groupby("ticker")[c].transform(
                lambda s: s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            )
    # Уровневые (перцентиль): ffill внутри тикера, остаток (самое начало) → 0.5
    # как «нейтральная медиана» (а не 0, что означало бы «минимум за год»).
    for c in _FFILL_THEN_ZERO:
        if c in df.columns:
            df[c] = df.groupby("ticker")[c].transform(
                lambda s: s.replace([np.inf, -np.inf], np.nan).ffill()
            )
            df[c] = df[c].fillna(0.5)
    # volume_z_score мог дать NaN из-за нулевого std — добиваем нулём.
    if "volume_z_score" in df.columns:
        df["volume_z_score"] = df["volume_z_score"].fillna(0.0)

    # Категориальные для pytorch-forecasting должны быть СТРОКАМИ.
    df["ticker"] = df["ticker"].astype(str)
    df["day_of_week"] = df["day_of_week"].astype(str)
    df["is_weekend_ahead"] = df["is_weekend_ahead"].astype(str)

    return df.reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════════
#  Распределение признаков по типам TFT + конфигурация TimeSeriesDataSet
# ════════════════════════════════════════════════════════════════════════════════

FEATURE_SPEC: dict[str, list[str]] = {
    # Тикер — статическая категория (embedding инструмента в модели).
    "static_categoricals": ["ticker"],

    # KNOWN FUTURE категориальные: известны на горизонт прогноза вперёд.
    "time_varying_known_categoricals": ["day_of_week", "is_weekend_ahead"],

    # KNOWN FUTURE числовые: время и расстояние до экспирации детерминированы.
    "time_varying_known_reals": ["time_idx", "days_to_moex_expiration"],

    # PAST OBSERVED числовые: микроструктура + макро + стационарность.
    "time_varying_unknown_reals": [
        "log_return",            # Блок 4 — основной ценовой вход
        "atr_percentile_252",    # Блок 4
        "amihud_illiquidity",    # Блок 2
        "volume_z_score",        # Блок 2
        "vwap_distance",         # Блок 2
        "imoex_return",          # Блок 3
        "rgbi_yield_change",     # Блок 3
        "rvi_return",            # Блок 3
    ],
}


def make_timeseries_dataset(
    df: pd.DataFrame,
    target: str = "log_return",
    max_encoder_length: int = 30,
    max_prediction_length: int = 1,
    add_target_to_unknown: bool = True,
):
    """
    Собирает pytorch_forecasting.TimeSeriesDataSet с корректным распределением
    признаков (см. FEATURE_SPEC). Требует `pip install pytorch-forecasting`.

    target — целевая переменная. По умолчанию log_return (предсказываем
    стационарную доходность). Если предсказываете диапазон — передайте свою
    цель (например 'next_high_pct') и убедитесь, что она есть в df.

    Замечание: target НЕ дублируем в time_varying_unknown_reals (pytorch-
    forecasting добавляет цель в энкодер автоматически). Поэтому из списка
    unknown-реалов цель исключается.
    """
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    if target not in df.columns:
        raise ValueError(f"target '{target}' отсутствует в df")

    unknown_reals = [c for c in FEATURE_SPEC["time_varying_unknown_reals"] if c != target]

    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=target,
        group_ids=["ticker"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=FEATURE_SPEC["static_categoricals"],
        time_varying_known_categoricals=FEATURE_SPEC["time_varying_known_categoricals"],
        time_varying_known_reals=FEATURE_SPEC["time_varying_known_reals"],
        time_varying_unknown_reals=unknown_reals + [target] if add_target_to_unknown
        else unknown_reals,
        # Нормализация цели в разрезе тикера — масштабы инструментов разные.
        target_normalizer=GroupNormalizer(groups=["ticker"], transformation=None),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )


# ════════════════════════════════════════════════════════════════════════════════
#  Self-test
# ════════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    print("=" * 64)
    print("  feature_engineering.py self-test")
    print("=" * 64)
    ok = True

    # (1) Третий четверг — всегда четверг и в диапазоне 15..21.
    samples = [(2026, 3), (2026, 6), (2026, 9), (2026, 12), (2025, 6)]
    for y, mo in samples:
        d = third_thursday(y, mo)
        good = (d.weekday() == 3) and (15 <= d.day <= 21)
        print(f"  3rd Thursday {y}-{mo:02d} = {d} (dow={d.weekday()}, day={d.day}) "
              f"{'OK' if good else 'FAIL'}")
        ok &= good

    # (2) Дни до экспирации: накануне 3-го четверга июня 2026 = 1, в сам день = 0,
    #     на следующий день должны прыгнуть к сентябрьской экспирации.
    exp_jun = third_thursday(2026, 6)
    d0 = _days_to_next_expiration(exp_jun)
    d_prev = _days_to_next_expiration(exp_jun - dt.timedelta(days=1))
    d_next = _days_to_next_expiration(exp_jun + dt.timedelta(days=1))
    exp_sep = third_thursday(2026, 9)
    good = (d0 == 0) and (d_prev == 1) and (d_next == (exp_sep - (exp_jun + dt.timedelta(days=1))).days)
    print(f"  days_to_exp: at={d0} prev={d_prev} next={d_next} {'OK' if good else 'FAIL'}")
    ok &= good

    # (3) Синтетический полный прогон build_tft_features на 2 тикерах.
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=300)
    rows = []
    for tk in ("AAA", "BBB"):
        price = 100.0
        for dte in dates:
            ret = rng.normal(0, 0.02)
            price *= np.exp(ret)
            hi = price * (1 + abs(rng.normal(0, 0.01)))
            lo = price * (1 - abs(rng.normal(0, 0.01)))
            op = lo + (hi - lo) * rng.random()
            rows.append((dte, tk, op, hi, lo, price, rng.integers(1e4, 1e6)))
    df = pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low", "close", "volume"])

    macro_rows = [(dte, 3000 * np.exp(rng.normal(0, 0.01)),
                   120 + rng.normal(0, 0.2), 20 * np.exp(rng.normal(0, 0.05)))
                  for dte in dates]
    macro_df = pd.DataFrame(macro_rows, columns=["date", "IMOEX_close", "RGBI_close", "RVI_close"])

    feat = build_tft_features(df, macro_df)
    needed = (FEATURE_SPEC["time_varying_known_categoricals"]
              + FEATURE_SPEC["time_varying_known_reals"]
              + FEATURE_SPEC["time_varying_unknown_reals"])
    missing = [c for c in needed if c not in feat.columns]
    has_all = not missing
    print(f"  build: {len(feat)} строк, {feat['ticker'].nunique()} тикера; "
          f"missing={missing or 'нет'} {'OK' if has_all else 'FAIL'}")
    ok &= has_all

    # NaN не должно остаться в признаках после обработки.
    nan_counts = feat[needed].isna().sum()
    no_nan = int(nan_counts.sum()) == 0
    print(f"  NaN в признаках: {int(nan_counts.sum())} {'OK' if no_nan else 'FAIL'}")
    if not no_nan:
        print(nan_counts[nan_counts > 0])
    ok &= no_nan

    # time_idx монотонен и стартует с 0 на каждый тикер.
    starts = feat.groupby("ticker")["time_idx"].min().unique().tolist()
    mono = bool((feat.groupby("ticker")["time_idx"].diff().dropna() == 1).all())
    good = starts == [0] and mono
    print(f"  time_idx: starts={starts} monotonic_step1={mono} {'OK' if good else 'FAIL'}")
    ok &= good

    # перцентиль в [0,1]
    p = feat["atr_percentile_252"]
    good = bool((p >= 0).all() and (p <= 1).all())
    print(f"  atr_percentile_252 ∈ [0,1]: {'OK' if good else 'FAIL'} "
          f"(min={p.min():.3f} max={p.max():.3f})")
    ok &= good

    print(f"\n  ИТОГ: {'ALL PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
