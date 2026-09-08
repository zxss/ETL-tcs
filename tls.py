"""
Единая точка настройки TLS для всех сетевых клиентов проекта.

По умолчанию сертификаты брокера ПРОВЕРЯЮТСЯ (config.INVEST_TLS_VERIFY=1).
Отключить проверку можно только явным INVEST_TLS_VERIFY=0 — тогда при первом
создании контекста в лог уходит предупреждение "TLS verification DISABLED!".

Использование:
    urllib.request.urlopen(req, context=tls.ssl_context())  # urllib / stdlib
    aiohttp.TCPConnector(ssl=tls.aiohttp_ssl())             # aiohttp

Сертификат T-Invest подписан «Russian Trusted Root CA» (Минцифры), которого нет
в дефолтном хранилище Python — путь к PEM с этим корнем задаётся через
INVEST_CA_BUNDLE (см. config.INVEST_CA_BUNDLE).
"""
from __future__ import annotations

import logging
import os
import ssl

import config

logger = logging.getLogger("tls")

_warned = False


def verify_enabled(override: bool | None = None) -> bool:
    """True — сертификаты проверяются. override перекрывает config (флаг CLI)."""
    if override is not None:
        return bool(override)
    return bool(config.INVEST_TLS_VERIFY)


def _base_context() -> ssl.SSLContext:
    """Доверенное хранилище: системное + (если задан) INVEST_CA_BUNDLE.
    Бандл нужен для «Russian Trusted Root CA», которым подписан T-Invest.

    Важно: create_default_context(cafile=...) ЗАМЕНЯЕТ системные корни, поэтому
    бандл догружается отдельным load_verify_locations — тогда доверяем и
    публичным УЦ, и корню из бандла."""
    ctx = ssl.create_default_context()
    bundle = os.path.expanduser(getattr(config, "INVEST_CA_BUNDLE", "") or "")
    if bundle:
        if not os.path.isfile(bundle):
            raise ValueError(f"INVEST_CA_BUNDLE: файл не найден: {bundle}")
        ctx.load_verify_locations(cafile=bundle)
    return ctx


def _warn_once() -> None:
    global _warned
    if not _warned:
        logger.warning("TLS verification DISABLED!")
        _warned = True


def ssl_context(override: bool | None = None) -> ssl.SSLContext:
    """SSL-контекст для urllib/stdlib. Без проверки — только по явному флагу."""
    ctx = _base_context()
    if verify_enabled(override):
        return ctx
    _warn_once()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def aiohttp_ssl(override: bool | None = None) -> ssl.SSLContext | bool:
    """Значение для aiohttp.TCPConnector(ssl=...): контекст или False."""
    if verify_enabled(override):
        return _base_context()
    _warn_once()
    return False
