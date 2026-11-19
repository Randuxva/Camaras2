"""
Logger estructurado en formato JSON para el Agente ANPR Edge.
Sin dependencias externas - implementación propia basada en logging stdlib.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from typing import Any


# rtsp://usuario:clave@host → rtsp://***@host. Al cortarse el video, el mensaje de error traía la
# dirección completa de la cámara y la contraseña quedaba escrita en logs/agente.log.
_CREDENCIALES = re.compile(r"(?<=://)[^/@\s:]+:[^/@\s]+@")


def ocultar_credenciales(texto: str) -> str:
    return _CREDENCIALES.sub("***@", texto)


class _JSONFormatter(logging.Formatter):
    """Formateador que emite cada línea como un objeto JSON válido."""

    RESERVED = frozenset(
        ("msg", "message", "args", "levelname", "levelno", "pathname",
         "filename", "module", "funcName", "lineno", "exc_info", "exc_text",
         "stack_info", "created", "msecs", "relativeCreated", "thread",
         "threadName", "processName", "process", "name", "taskName")
    )

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        )[:-3] + "Z"

        obj: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "msg": ocultar_credenciales(record.getMessage()),
        }

        # Excepción
        if record.exc_info:
            obj["exc"] = [ocultar_credenciales(x) for x in traceback.format_exception(*record.exc_info)]

        # Campos extra que el caller pasó como keyword en extra={}
        for key, val in record.__dict__.items():
            if key not in self.RESERVED and not key.startswith("_"):
                try:
                    json.dumps(val)   # Solo serializables
                    obj[key] = ocultar_credenciales(val) if isinstance(val, str) else val
                except (TypeError, ValueError):
                    obj[key] = ocultar_credenciales(str(val))

        return json.dumps(obj, ensure_ascii=False)


_FILE_HANDLER: logging.Handler | None = None
_FILE_HANDLER_TRIED = False


def _file_handler() -> logging.Handler | None:
    """
    Archivo con rotación (5 × 5 MB), compartido por todos los loggers. Sin esto, cuando el agente
    falló durante días (octubre de 2026) no quedó ningún rastro del error: solo se veía en la consola.
    """
    global _FILE_HANDLER, _FILE_HANDLER_TRIED
    if _FILE_HANDLER_TRIED:
        return _FILE_HANDLER
    _FILE_HANDLER_TRIED = True
    path = os.getenv("LOG_FILE")
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "agente.log")
    if not path:
        return None
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        h = logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
        h.setFormatter(_JSONFormatter())
        _FILE_HANDLER = h
    except OSError:
        _FILE_HANDLER = None
    return _FILE_HANDLER


def _build_logger(name: str, level: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Ya inicializado

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter())
    logger.addHandler(handler)
    file_handler = _file_handler()
    if file_handler:
        logger.addHandler(file_handler)
    logger.propagate = False

    # Silenciar loggers ruidosos de terceros
    for noisy in ("ultralytics", "urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger


# ── API pública ─────────────────────────────────────────────────────────────

def get_logger(name: str = "anpr_edge") -> logging.Logger:
    """
    Devuelve un logger JSON configurado.

    Uso:
        log = get_logger(__name__)
        log.info("Vehículo detectado", extra={"track_id": 7, "direction": "ENTRADA"})
    """
    try:
        from config import cfg
        level = cfg.log_level
    except Exception:
        level = "INFO"
    return _build_logger(name, level)
