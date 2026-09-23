#!/usr/bin/env python3
"""
Точка входа ETL-приложения.

Запуск:
    python3 main.py

Переменные окружения (или .env):
    INVEST_TOKEN  — токен T-Инвестиций
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD — параметры PostgreSQL

Логика:
  - Проверяет наличие данных в БД для каждого тикера.
  - Скачивает только новые данные (инкрементальный режим).
  - Все тикеры загружаются параллельно (asyncio + aiohttp).

СТАТУС КОНВЕЙЕРА
----------------
Каждый этап выполняется под учётом статуса (SUCCESS / FAILED / SKIPPED /
WARNING). Раньше семь подряд идущих `except Exception` глушили любую ошибку, и
скрипт всегда возвращал 0 — упавший прогноз выглядел как успешный прогон.
Теперь в конце печатается сводка, а падение любого этапа из CRITICAL_STAGES
даёт sys.exit(1), чтобы cron/CI это заметили.
"""

import logging
import sys
from contextlib import contextmanager

import database
from services.load_history import run
from services import run_validation
from services import backfill_daily
from services import load_index, load_instruments
from services.data_freshness import StaleDataError, ensure_fresh_data
from services import calendar as trading_calendar_svc
import config
import tft_forecast

# Этапы, падение которых делает весь прогон неуспешным (exit code 1).
CRITICAL_STAGES = ["etl", "forecast"]

SUCCESS, FAILED, SKIPPED, WARNING = "SUCCESS", "FAILED", "SKIPPED", "WARNING"


class PipelineStatus:
    """Учёт статусов этапов конвейера и итоговая сводка."""

    def __init__(self, log: logging.Logger):
        self.log = log
        self._order: list[str] = []
        self._state: dict[str, tuple[str, str, str]] = {}   # key -> (label, status, note)

    def _set(self, key: str, label: str, status: str, note: str = "") -> None:
        if key not in self._state:
            self._order.append(key)
        self._state[key] = (label, status, note)

    def skip(self, key: str, label: str, note: str = "") -> None:
        self._set(key, label, SKIPPED, note)

    def warn(self, key: str, label: str, note: str = "") -> None:
        self._set(key, label, WARNING, note)

    @contextmanager
    def stage(self, key: str, label: str):
        """Выполняет этап, ловит исключение и помечает его FAILED.

        Исключение наружу не выпускается: конвейер идёт дальше, чтобы дать
        полную картину в сводке. Итоговый код возврата решает failed_critical().
        """
        self._set(key, label, SUCCESS)
        try:
            yield self
        except Exception as e:  # noqa: BLE001 — статус этапа важнее прерывания
            note = f"{type(e).__name__}: {e}".strip()
            if len(note) > 160:
                note = note[:157] + "..."
            self._set(key, label, FAILED, note)
            self.log.exception("Этап %s завершился с ошибкой: %s", label, e)

    def failed_critical(self) -> list[str]:
        return [self._state[k][0] for k in self._order
                if k in CRITICAL_STAGES and self._state[k][1] == FAILED]

    def render(self) -> None:
        self.log.info("")
        self.log.info("Pipeline Summary:")
        for key in self._order:
            label, status, note = self._state[key]
            line = f"- {label}: {status}"
            if note:
                line += f" ({note})"
            if status == FAILED:
                self.log.error(line)
            elif status == WARNING:
                self.log.warning(line)
            else:
                self.log.info(line)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("main")
    st = PipelineStatus(log)

    log.info("Подключение к PostgreSQL...")
    try:
        conn = database.get_connection()
        database.init_db(conn)
    except Exception as e:  # noqa: BLE001 — без схемы работать нечем
        log.error("Инициализация БД не удалась: %s", e)
        log.info("")
        log.info("Pipeline Summary:")
        log.error("- Database: FAILED (%s)", e)
        sys.exit(1)

    try:
        with st.stage("etl", "ETL"):
            run(conn)

        # Справочник инструментов (лотность, шаг цены, доступность шорта).
        # Нужен слою ликвидности: market_data.volume приходит В ЛОТАХ, и без
        # размера лота рублёвый оборот занижается — для TGKA в 100 000 раз.
        with st.stage("instruments", "Instruments"):
            load_instruments.run(conn)

        # Индексы (IMOEX) — отдельный шаг для слоя рыночного контекста.
        log.info("Загрузка биржевых индексов (IMOEX)...")
        with st.stage("index", "Index"):
            load_index.run(conn)

        # Достраиваем дневные свечи из 5M, если официальный дневной бар ещё
        # не опубликован брокером (вечер/ночь). Самокорректируется позже.
        with st.stage("backfill", "Backfill 1D←5M"):
            backfill_daily.backfill(conn, config.TICKERS)

        # Календарь биржи: подтягиваем расписание MOEX на ближайшие 2 недели.
        # Сбой сети/API не критичен — TradingCalendar прозрачно остаётся на
        # встроенном оффлайн-списке праздников и переносов.
        with st.stage("calendar", "Trading calendar"):
            if trading_calendar_svc.sync_schedule():
                log.info("Календарь биржи синхронизирован с MOEX.")
            else:
                st.warn("calendar", "Trading calendar",
                        "API недоступен — оффлайн-календарь")

        # Гард свежести: свечи уже догружены выше (refresh=False — только проверка).
        # Дашборд деньги не двигает, поэтому НЕ падаем — громко предупреждаем, что
        # прогноз пойдёт на устаревших данных (в отличие от place_orders, который
        # на этом блокирует торговлю).
        with st.stage("freshness", "Freshness"):
            try:
                ensure_fresh_data(conn, refresh=False)
            except StaleDataError as e:
                log.warning("!" * 76)
                log.warning("ВНИМАНИЕ: %s", e)
                log.warning("Дашборд и прогноз считаются на УСТАРЕВШИХ дневных свечах.")
                log.warning("!" * 76)
                st.warn("freshness", "Freshness", "устаревшие дневные свечи")

        combined = getattr(config, "COMBINED_TABLE", True)

        log.info("Данные загружены — запуск статистической валидации стратегий...")
        val_rows = None
        if getattr(config, "RUN_VALIDATION", True):
            with st.stage("validation", "Validation"):
                val_rows = run_validation.run(conn, quiet=combined)
        else:
            st.skip("validation", "Validation", "RUN_VALIDATION=0")

        log.info("Запуск Multi-Asset TFT — прогноз диапазона на следующий день...")
        forecasts = None
        if getattr(config, "TFT_FORECAST", True):
            with st.stage("forecast", "Forecast"):
                forecasts = tft_forecast.run(conn, quiet=combined)
        else:
            st.skip("forecast", "Forecast", "TFT_FORECAST=0")

        if combined:
            with st.stage("dashboard", "Dashboard"):
                # Вселенная дашборда — все бумаги с прогнозом (а не только
                # VALIDATION_TICKERS), чтобы вывести топ-N по всему рынку.
                fc = forecasts or {}
                universe = [k for k in fc if k != "__meta__"] or config.VALIDATION_TICKERS
                tft_forecast.print_combined(
                    val_rows, forecasts,
                    universe, config.VALIDATION_STRATS,
                    show_all=getattr(config, "SHOW_ALL_INTRADAY", False),
                    top_n=getattr(config, "DASHBOARD_TOP_N", 50),
                )
        else:
            st.skip("dashboard", "Dashboard", "COMBINED_TABLE=0")

        # Недельный дашборд (формат дневного): рекомендация LONG/SHORT из
        # 5-дневных квантилей, окно прогноза от текущего дня недели.
        if forecasts:
            with st.stage("weekly", "Weekly Dashboard"):
                tft_forecast.print_weekly_dashboard(
                    forecasts, top_n=getattr(config, "DASHBOARD_TOP_N", 50))
        else:
            st.skip("weekly", "Weekly Dashboard", "нет прогноза")

        # Выставление заявок — отдельная команда (services.place_orders),
        # main.py деньги не двигает. В сводке показываем это явно.
        st.skip("place_orders", "Place Orders", "отдельная команда: python3 -m services.place_orders")
    finally:
        conn.close()
        database.close_pool()

    st.render()

    failed = st.failed_critical()
    if failed:
        log.error("Критические этапы упали: %s — выход с кодом 1.", ", ".join(failed))
        sys.exit(1)

    log.info("ETL + валидация завершены.")


if __name__ == "__main__":
    main()
