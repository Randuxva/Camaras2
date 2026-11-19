"""
Prioridad del hilo actual. La detección (YOLO) es lo único que no puede atrasarse: si se frena,
un vehículo cruza el portón entre dos cuadros analizados. El decodificador del stream principal
y la lectura de patentes pueden esperar.

Medido en la grabación del 2026-10-02: cuando la escena tiene mucho movimiento, decodificar el
principal (2880x1620 H.265) acaparaba los 2 núcleos y YOLO pasaba de ~0,2 s a 1–2 s por cuadro;
se perdió una moto y la salida de un auto se partió en dos eventos.
"""
from __future__ import annotations

import os
import sys

from utils.logger import get_logger

log = get_logger(__name__)


def lower_current_thread_priority() -> None:
    """Baja la prioridad del hilo que la llama (Windows: BELOW_NORMAL; Linux: nice +10)."""
    try:
        if sys.platform == "win32":
            import ctypes

            THREAD_PRIORITY_BELOW_NORMAL = -1
            k32 = ctypes.windll.kernel32
            k32.SetThreadPriority(k32.GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL)
        elif hasattr(os, "setpriority") and hasattr(os, "gettid"):
            os.setpriority(os.PRIO_PROCESS, os.gettid(), 10)
    except Exception as e:  # nunca debe impedir que el hilo trabaje
        log.debug(f"No se pudo bajar la prioridad del hilo: {e}")
