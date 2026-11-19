"""
Estado de tracks con TTL y cooldown por patente.

Responsabilidades:
  1. Mantener un registro de track IDs ya procesados (con TTL para evitar memory leaks
     y permitir re-proceso si el mismo ID es reciclado por YOLO tras mucho tiempo).
  2. Cooldown por patente+dirección: si "ABC123_ENTRADA" fue procesado hace < N segundos,
     ignorar nueva detección aunque sea un track_id diferente.

Thread-safety: pensado para uso en thread único (consumer). Si se migra a multi-thread,
agregar threading.Lock en los métodos de escritura.
"""
from __future__ import annotations

import time
from typing import Dict, Tuple

from utils.logger import get_logger

log = get_logger(__name__)


class TrackerState:
    """
    Gestión de IDs de tracks procesados y cooldowns por patente.

    processed_ids  : dict[track_id → timestamp_procesado]
                     (TTL configurable, evita memory leak y reciclaje de IDs)

    plate_cooldown : dict[(patente, dirección) → timestamp_procesado]
                     (previene re-registro del mismo vehículo dentro del cooldown)
    """

    def __init__(
        self,
        track_id_ttl_sec: int = 300,    # 5 minutos
        plate_cooldown_sec: int = 45,   # 45 segundos por patente+dirección
        max_active: int = 20,           # Límite de tracks simultáneos
    ) -> None:
        self._track_id_ttl  = float(track_id_ttl_sec)
        self._plate_cooldown = float(plate_cooldown_sec)
        self._max_active    = max_active

        # track_id → timestamp cuando fue procesado
        self._processed: Dict[int, float] = {}
        # (plate, direction) → timestamp cuando fue procesado
        self._plate_cd: Dict[Tuple[str, str], float] = {}

    # ── Consultas ──────────────────────────────────────────────────────────

    def is_track_processed(self, track_id: int) -> bool:
        """True si el track_id fue procesado recientemente (dentro del TTL)."""
        ts = self._processed.get(track_id)
        if ts is None:
            return False
        if time.monotonic() - ts > self._track_id_ttl:
            self._processed.pop(track_id, None)
            return False
        return True

    def is_plate_in_cooldown(self, plate: str, direction: str) -> bool:
        """
        True si la misma patente+dirección fue registrada dentro del cooldown.
        Ignora placas vacías/desconocidas para no bloquear registros manuales.
        """
        if not plate or plate.startswith("DESC-"):
            return False
        key = (plate.upper(), direction)
        ts = self._plate_cd.get(key)
        if ts is None:
            return False
        if time.monotonic() - ts > self._plate_cooldown:
            self._plate_cd.pop(key, None)
            return False
        return True

    # ── Mutaciones ──────────────────────────────────────────────────────────

    def mark_track_processed(self, track_id: int) -> None:
        """Marca el track_id como procesado con timestamp actual."""
        self._processed[track_id] = time.monotonic()

    def mark_plate_processed(self, plate: str, direction: str) -> None:
        """Registra el cooldown para patente+dirección."""
        if plate and not plate.startswith("DESC-"):
            self._plate_cd[(plate.upper(), direction)] = time.monotonic()

    # ── Mantenimiento ──────────────────────────────────────────────────────

    def cleanup(self) -> Tuple[int, int]:
        """
        Elimina entradas vencidas. Llamar periódicamente (ej. cada 60 frames).
        Retorna (tracks_removidos, cooldowns_removidos).
        """
        now = time.monotonic()

        stale_tracks = [
            tid for tid, ts in self._processed.items()
            if now - ts > self._track_id_ttl
        ]
        for tid in stale_tracks:
            del self._processed[tid]

        stale_plates = [
            key for key, ts in self._plate_cd.items()
            if now - ts > self._plate_cooldown
        ]
        for key in stale_plates:
            del self._plate_cd[key]

        if stale_tracks or stale_plates:
            log.debug(
                "TrackerState cleanup",
                extra={"tracks_removed": len(stale_tracks), "cooldowns_removed": len(stale_plates)},
            )

        return len(stale_tracks), len(stale_plates)

    @property
    def processed_count(self) -> int:
        return len(self._processed)

    @property
    def cooldown_count(self) -> int:
        return len(self._plate_cd)
