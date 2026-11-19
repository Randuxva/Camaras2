"""
Cola persistente SQLite para modo offline y reintentos en el Agente ANPR Edge (2026).
Garantiza idempotencia y cero pérdida de datos ante caídas de conexión a Internet.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from utils.logger import get_logger

log = get_logger(__name__)

# Espera entre reintentos: 15 s, 30 s, 1 min ... hasta 30 min. Un evento NO se descarta por
# cantidad de intentos (así se perdieron 226 pasos por el portón en octubre de 2026 cuando la
# función respondió error durante horas): solo se da por perdido si pasa MAX_AGE sin subir.
RETRY_BASE_SEC = 15.0
RETRY_MAX_SEC = 1800.0


def retry_delay(attempts: int) -> float:
    return min(RETRY_BASE_SEC * (2 ** min(max(attempts - 1, 0), 10)), RETRY_MAX_SEC)


@dataclass
class QueuedEvent:
    id: str
    track_id: int
    event_type: str        # "ENTRADA" | "SALIDA"
    image_path: str        # Ruta local de la imagen crop
    timestamp: int         # Epoch timestamp
    attempts: int
    created_at: float
    camera_id: str
    sucursal_id: str
    extra_paths: List[str] = field(default_factory=list)  # recortes adicionales (lectura multi-frame)
    reason: str = ""                                    # motivo del evento en el ROI engine
    local_ocr: Optional[dict] = None                    # lectura local de la patente (core/plate_reader.py)
    vehicle_class: str = ""                             # clase del detector: car | motorcycle | bus | truck
    crop_meta: Optional[dict] = None                    # calidad del recorte principal: score, rejilla, truncated, lighting, sinColor


class OfflineQueue:
    """
    Gestiona una base de datos SQLite embebida local para almacenar eventos
    cuando falla la conexión a Supabase/Webhook.
    """

    def __init__(self, db_path: str = "offline_queue.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pending_events (
                        id TEXT PRIMARY KEY,
                        track_id INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        image_path TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        attempts INTEGER DEFAULT 0,
                        created_at REAL NOT NULL,
                        camera_id TEXT NOT NULL,
                        sucursal_id TEXT NOT NULL,
                        status TEXT DEFAULT 'PENDING'
                    );
                """)
                # Migración de columnas nuevas en bases SQLite ya existentes
                cols = {r["name"] for r in conn.execute("PRAGMA table_info(pending_events)")}
                if "extra_paths" not in cols:
                    conn.execute("ALTER TABLE pending_events ADD COLUMN extra_paths TEXT DEFAULT '[]'")
                if "reason" not in cols:
                    conn.execute("ALTER TABLE pending_events ADD COLUMN reason TEXT DEFAULT ''")
                if "local_ocr" not in cols:
                    conn.execute("ALTER TABLE pending_events ADD COLUMN local_ocr TEXT DEFAULT ''")
                if "vehicle_class" not in cols:
                    conn.execute("ALTER TABLE pending_events ADD COLUMN vehicle_class TEXT DEFAULT ''")
                if "crop_meta" not in cols:
                    conn.execute("ALTER TABLE pending_events ADD COLUMN crop_meta TEXT DEFAULT ''")
                if "next_attempt_at" not in cols:
                    conn.execute("ALTER TABLE pending_events ADD COLUMN next_attempt_at REAL DEFAULT 0")
                if "last_error" not in cols:
                    conn.execute("ALTER TABLE pending_events ADD COLUMN last_error TEXT DEFAULT ''")
                # Tramos en que el portón quedó sin vigilancia (ver cloud/watch_gaps.py)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS watch_gaps (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        desde REAL NOT NULL,
                        hasta REAL NOT NULL,
                        motivo TEXT NOT NULL,
                        sent INTEGER DEFAULT 0
                    );
                """)
                conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL);")
                # Libro de eventos: TODO paso por el portón que detectó el agente y qué pasó con él.
                # pending_events se vacía al subir; esto queda como constancia para auditar
                # (tools/auditoria.py) y contrastar contra la base.
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS event_log (
                        id TEXT PRIMARY KEY,
                        track_id INTEGER,
                        event_type TEXT NOT NULL,
                        reason TEXT DEFAULT '',
                        timestamp INTEGER NOT NULL,
                        created_at REAL NOT NULL,
                        camera_id TEXT,
                        n_images INTEGER DEFAULT 0,
                        local_ocr TEXT DEFAULT '',
                        status TEXT DEFAULT 'EN_COLA',
                        attempts INTEGER DEFAULT 0,
                        uploaded_at REAL,
                        action TEXT,
                        movimiento_codigo INTEGER,
                        plate TEXT,
                        response TEXT DEFAULT '',
                        last_error TEXT DEFAULT ''
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_event_log_ts ON event_log(timestamp);")
                # Lo que ya estaba en cola (o dado por perdido) antes de existir el libro
                conn.execute("""
                    INSERT OR IGNORE INTO event_log (id, track_id, event_type, reason, timestamp, created_at, camera_id, local_ocr, status, attempts)
                    SELECT id, track_id, event_type, reason, timestamp, created_at, camera_id, local_ocr,
                           CASE status WHEN 'FAILED' THEN 'FALLIDO' ELSE 'EN_COLA' END, attempts
                    FROM pending_events
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON pending_events(status);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON pending_events(created_at);")
                conn.commit()
        except Exception as e:
            log.error(f"Error inicializando base de datos SQLite offline: {e}")

    def enqueue(
        self,
        track_id: int,
        event_type: str,
        image_path: str,
        timestamp: int,
        camera_id: str,
        sucursal_id: str,
        event_id: Optional[str] = None,
        extra_paths: Optional[List[str]] = None,
        reason: str = "",
        local_ocr: Optional[dict] = None,
        vehicle_class: str = "",
        crop_meta: Optional[dict] = None,
    ) -> str:
        eid = event_id or str(uuid.uuid4())
        now = time.time()
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT INTO pending_events (
                        id, track_id, event_type, image_path, timestamp, attempts, created_at, camera_id, sucursal_id, status,
                        extra_paths, reason, local_ocr, vehicle_class, crop_meta
                    ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?)
                """, (eid, track_id, event_type, image_path, timestamp, now, camera_id, sucursal_id,
                      json.dumps(extra_paths or []), reason, json.dumps(local_ocr) if local_ocr else "", vehicle_class or "",
                      json.dumps(crop_meta) if crop_meta else ""))
                conn.execute("""
                    INSERT OR IGNORE INTO event_log (id, track_id, event_type, reason, timestamp, created_at, camera_id, n_images, local_ocr)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (eid, track_id, event_type, reason, timestamp, now, camera_id, 1 + len(extra_paths or []),
                      json.dumps(local_ocr) if local_ocr else ""))
                conn.commit()
            log.info(f"Evento encolado en SQLite local: {eid} ({event_type})")
            return eid
        except Exception as e:
            log.error(f"Error encolando evento {eid}: {e}")
            return eid

    def get_pending(self, limit: int = 10, due_only: bool = True) -> List[QueuedEvent]:
        """Eventos por subir, del más viejo al más nuevo. due_only: solo los que ya cumplieron su espera."""
        try:
            with self._get_conn() as conn:
                cursor = conn.execute("""
                    SELECT id, track_id, event_type, image_path, timestamp, attempts, created_at, camera_id, sucursal_id,
                           extra_paths, reason, local_ocr, vehicle_class, crop_meta
                    FROM pending_events
                    WHERE status = 'PENDING' AND COALESCE(next_attempt_at, 0) <= ?
                    ORDER BY created_at ASC
                    LIMIT ?
                """, (time.time() if due_only else float("inf"), limit))
                rows = cursor.fetchall()
                return [
                    QueuedEvent(
                        id=row["id"],
                        track_id=row["track_id"],
                        event_type=row["event_type"],
                        image_path=row["image_path"],
                        timestamp=row["timestamp"],
                        attempts=row["attempts"],
                        created_at=row["created_at"],
                        camera_id=row["camera_id"],
                        sucursal_id=row["sucursal_id"],
                        extra_paths=json.loads(row["extra_paths"] or "[]"),
                        reason=row["reason"] or "",
                        local_ocr=json.loads(row["local_ocr"]) if row["local_ocr"] else None,
                        vehicle_class=row["vehicle_class"] or "",
                        crop_meta=json.loads(row["crop_meta"]) if row["crop_meta"] else None,
                    )
                    for row in rows
                ]
        except Exception as e:
            log.error(f"Error obteniendo eventos pendientes de SQLite: {e}")
            return []

    def mark_completed(self, event_id: str, result: Optional[dict] = None, status: str = "SUBIDO") -> None:
        """Saca el evento de la cola y deja en el libro qué respondió la nube."""
        try:
            with self._get_conn() as conn:
                data = (result or {}).get("data") or {}
                mov = data.get("movimiento") or {}
                veh = data.get("vehiculo")
                conn.execute("""
                    UPDATE event_log SET status = ?, uploaded_at = ?, action = ?, movimiento_codigo = ?, plate = ?, response = ?
                    WHERE id = ?
                """, (status, time.time(), (result or {}).get("action"), mov.get("codigo") or data.get("codigo"),
                      veh if isinstance(veh, str) else mov.get("placaOCR"),
                      json.dumps(result, ensure_ascii=False, default=str)[:2000] if result else "", event_id))
                # Obtenemos la ruta para borrar el archivo temporal si existe
                cursor = conn.execute("SELECT image_path, extra_paths FROM pending_events WHERE id = ?", (event_id,))
                row = cursor.fetchone()
                if row:
                    paths = [row["image_path"]] + json.loads(row["extra_paths"] or "[]")
                    for path in paths:
                        if path and os.path.exists(path):
                            try:
                                os.remove(path)
                            except Exception as err:
                                log.warning(f"No se pudo borrar archivo local {path}: {err}")

                conn.execute("DELETE FROM pending_events WHERE id = ?", (event_id,))
                conn.commit()
            log.debug(f"Evento {event_id} completado y eliminado de la cola local.")
        except Exception as e:
            log.error(f"Error marcando evento {event_id} como completado: {e}")

    def increment_attempts(self, event_id: str, max_age_sec: float = 14 * 86400, error: str = "") -> None:
        """
        Registra un intento fallido y programa el próximo con espera creciente. El evento solo
        pasa a FAILED si lleva más de max_age_sec sin poder subirse.
        """
        try:
            with self._get_conn() as conn:
                row = conn.execute("SELECT attempts, created_at FROM pending_events WHERE id = ?", (event_id,)).fetchone()
                if not row:
                    return
                attempts = row["attempts"] + 1
                now = time.time()
                if now - row["created_at"] > max_age_sec:
                    conn.execute("UPDATE pending_events SET status = 'FAILED', attempts = ?, last_error = ? WHERE id = ?",
                                 (attempts, error[:300], event_id))
                    conn.execute("UPDATE event_log SET status = 'FALLIDO', attempts = ?, last_error = ? WHERE id = ?",
                                 (attempts, error[:300], event_id))
                    log.error(f"Evento {event_id} lleva más de {max_age_sec / 86400:.0f} días sin subir. Marcado FAILED.")
                else:
                    conn.execute("UPDATE pending_events SET attempts = ?, next_attempt_at = ?, last_error = ? WHERE id = ?",
                                 (attempts, now + retry_delay(attempts), error[:300], event_id))
                    conn.execute("UPDATE event_log SET attempts = ?, last_error = ? WHERE id = ?", (attempts, error[:300], event_id))
                conn.commit()
        except Exception as e:
            log.error(f"Error incrementando intentos para evento {event_id}: {e}")

    def requeue_failed(self, max_age_sec: float = 45 * 86400, solo_con_patente: bool = False) -> int:
        """
        Vuelve a poner en cola los eventos FAILED que todavía tienen su foto. Devuelve cuántos.
        solo_con_patente: deja afuera los que el portón no pudo leer (reenviados días después
        solo suman registros "para revisar" que ya nadie puede resolver).
        """
        n = 0
        try:
            with self._get_conn() as conn:
                rows = conn.execute("SELECT id, image_path, created_at, local_ocr FROM pending_events WHERE status = 'FAILED'").fetchall()
                for r in rows:
                    if time.time() - r["created_at"] > max_age_sec or not os.path.exists(r["image_path"]):
                        continue
                    if solo_con_patente and not ((json.loads(r["local_ocr"]) if r["local_ocr"] else None) or {}).get("plate"):
                        continue
                    conn.execute("UPDATE pending_events SET status = 'PENDING', attempts = 0, next_attempt_at = 0 WHERE id = ?", (r["id"],))
                    conn.execute("UPDATE event_log SET status = 'EN_COLA' WHERE id = ?", (r["id"],))
                    n += 1
                conn.commit()
        except Exception as e:
            log.error(f"Error reencolando eventos fallidos: {e}")
        return n

    def event_log_rows(self, desde: float = 0.0, hasta: float = float("inf")) -> List[dict]:
        """Libro de eventos entre dos instantes (epoch), del más viejo al más nuevo."""
        try:
            with self._get_conn() as conn:
                rows = conn.execute("SELECT * FROM event_log WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp", (desde, hasta)).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            log.error(f"Error leyendo el libro de eventos: {e}")
            return []

    def accion_previa(self, patente: str, event_type: str, desde: float, hasta: float) -> Optional[tuple]:
        """
        Resultado en la nube del aviso MÁS RECIENTE de esa patente y ese sentido entre dos instantes (epoch):
        (acción, timestamp del evento) o None si no hay ninguno ya respondido. Solo lee: no cambia nada.
        """
        try:
            with self._get_conn() as conn:
                r = conn.execute(
                    "SELECT action, timestamp FROM event_log WHERE event_type = ? AND timestamp >= ? AND timestamp <= ? "
                    "AND action IS NOT NULL AND local_ocr LIKE ? ORDER BY timestamp DESC LIMIT 1",
                    (event_type, desde, hasta, f'%"plate": "{patente}"%')).fetchone()
                return (r["action"], r["timestamp"]) if r else None
        except Exception as e:
            log.error(f"Error leyendo el resultado previo de una patente: {e}")
            return None

    def count_failed(self) -> int:
        try:
            with self._get_conn() as conn:
                row = conn.execute("SELECT count(*) AS n FROM pending_events WHERE status = 'FAILED'").fetchone()
                return int(row["n"]) if row else 0
        except Exception:
            return 0

    # ── Huecos de vigilancia ────────────────────────────────────────────────
    def kv_get(self, key: str) -> Optional[str]:
        try:
            with self._get_conn() as conn:
                row = conn.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
                return row["v"] if row else None
        except Exception:
            return None

    def kv_set(self, key: str, value: str) -> None:
        try:
            with self._get_conn() as conn:
                conn.execute("INSERT INTO kv (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v", (key, value))
                conn.commit()
        except Exception as e:
            log.debug(f"kv_set({key}) falló: {e}")

    def add_gap(self, desde: float, hasta: float, motivo: str) -> None:
        try:
            with self._get_conn() as conn:
                conn.execute("INSERT INTO watch_gaps (desde, hasta, motivo) VALUES (?, ?, ?)", (desde, hasta, motivo))
                conn.commit()
        except Exception as e:
            log.error(f"No se pudo guardar el hueco de vigilancia: {e}")

    def unsent_gaps(self, limit: int = 20) -> List[dict]:
        try:
            with self._get_conn() as conn:
                rows = conn.execute("SELECT id, desde, hasta, motivo FROM watch_gaps WHERE sent = 0 ORDER BY desde LIMIT ?", (limit,)).fetchall()
                return [{"id": r["id"], "from": r["desde"], "to": r["hasta"], "reason": r["motivo"]} for r in rows]
        except Exception:
            return []

    def mark_gaps_sent(self, ids: List[int]) -> None:
        if not ids:
            return
        try:
            with self._get_conn() as conn:
                conn.executemany("UPDATE watch_gaps SET sent = 1 WHERE id = ?", [(i,) for i in ids])
                conn.commit()
        except Exception as e:
            log.error(f"No se pudieron marcar los huecos como enviados: {e}")

    def count_pending(self) -> int:
        try:
            with self._get_conn() as conn:
                cursor = conn.execute("SELECT count(*) as count FROM pending_events WHERE status = 'PENDING'")
                row = cursor.fetchone()
                return int(row["count"]) if row else 0
        except Exception:
            return 0
