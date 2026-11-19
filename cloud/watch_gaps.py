"""
Huecos de vigilancia: tramos en que el portón quedó sin mirar.

Entre el 6 y el 9 de octubre de 2026 la PC del portón se suspendió unas 22 horas y nadie lo supo:
todo auto que pasó en esos tramos no existe para el sistema (después aparece como "salida sin
entrada" o queda "adentro" para siempre). El agente es el único que sabe con certeza cuándo no
estuvo mirando, así que lo anota acá y lo informa en el latido. El panel lo muestra y el cobro
automático no se aplica a estadías que cruzan un hueco.

Un corte de internet NO es un hueco: el agente sigue registrando y sube los eventos después.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from utils.logger import get_logger

log = get_logger(__name__)

KEY_LAST_OK = "watch_last_ok"
# Menos que esto no se informa (reconexiones normales de la cámara, reinicio rápido del agente)
MIN_GAP_SEC = 180.0
# Si el proceso estuvo sin ejecutar más que esto, la PC estuvo suspendida o el agente apagado
PROCESS_STALL_SEC = 90.0


class WatchGapTracker:
    """Se llama cada pocos segundos con `watching` = la cámara está entregando imagen."""

    def __init__(self, store, clock: Callable[[], float] = time.time) -> None:
        self._store = store
        self._clock = clock  # reloj de pared: el monotónico puede no avanzar durante la suspensión
        self._last_tick: Optional[float] = None
        self._stalled = False
        raw = store.kv_get(KEY_LAST_OK)
        try:
            self._last_ok: Optional[float] = float(raw) if raw else None
        except ValueError:
            self._last_ok = None

    def tick(self, watching: bool) -> Optional[dict]:
        """Devuelve el hueco registrado en este paso, si lo hubo."""
        now = self._clock()
        if self._last_tick is None or now - self._last_tick > PROCESS_STALL_SEC:
            # Primer paso tras arrancar, o el proceso estuvo detenido (suspensión)
            self._stalled = True
        self._last_tick = now
        if not watching:
            return None

        gap = None
        if self._last_ok is not None and now - self._last_ok >= MIN_GAP_SEC:
            motivo = "SIN_REPORTE" if self._stalled else "SIN_VIDEO"
            gap = {"from": self._last_ok, "to": now, "reason": motivo}
            self._store.add_gap(self._last_ok, now, motivo)
            log.warning(f"El portón estuvo sin vigilancia {int((now - self._last_ok) / 60)} min ({motivo}).")
        self._stalled = False
        self._last_ok = now
        self._store.kv_set(KEY_LAST_OK, repr(now))
        return gap
