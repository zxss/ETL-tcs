"""
Multi-Asset TFT Next-Day Range Forecast.

Единая модель Temporal Fusion Transformer, обучаемая одновременно на ВСЕХ
доступных тикерах, для прогноза диапазона цены следующего торгового дня:

    ForecastLow, ForecastHigh, ExpectedRangePct, CoverageProb

Публичная точка входа — forecast.run(conn) — вызывается из main.py после
загрузки данных и валидации стратегий.

Модуль самодостаточен:
  * если установлен PyTorch — обучается настоящий TFT (variable selection,
    static ticker embeddings, LSTM-энкодер, interpretable multi-head attention,
    квантильные головы);
  * если torch недоступен — используется калиброванный статистический baseline
    (EWMA-волатильность + эмпирическое покрытие), чтобы main.py не падал.
"""

from .forecast import run  # noqa: F401
from .combined import print_combined  # noqa: F401

__all__ = ["run", "print_combined"]
