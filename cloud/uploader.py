"""
Módulo de subida de imágenes a Supabase Storage y disparo de Webhooks.
Integra compresión WebP, reintentos con cola offline (SQLite), envío de varios
recortes por evento (lectura multi-frame) y latido periódico hacia Supabase.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, List, Optional

import cv2
import numpy as np
import requests
from supabase import Client, create_client

from cloud.offline_queue import OfflineQueue, QueuedEvent
from cloud.watch_gaps import WatchGapTracker
from config import BASE_DIR, cfg
from utils.logger import get_logger

log = get_logger(__name__)

BUCKET = "anpr-captures"
CAPTURES_DIR = BASE_DIR / "captures"
AGENT_VERSION = "2026.10.2"


class CloudUploader:
    """
    Gestiona la subida de recortes a Supabase Storage y notificaciones Webhook.
    Ejecuta un worker en segundo plano para procesar la cola de reintentos offline.
    """

    def __init__(self, offline_queue: OfflineQueue) -> None:
        self.offline_queue = offline_queue
        self.supabase: Optional[Client] = None
        self._init_supabase()

        # Eventos que se están procesando ahora: evita que el envío inmediato y el
        # worker de reintentos suban/notifiquen el mismo evento en paralelo.
        self._in_flight: set[str] = set()
        self._in_flight_lock = threading.Lock()
        self._last_heartbeat = 0.0
        self._last_status = 0.0
        # Lo asigna main.py: devuelve el estado de la cámara (camera_ok, last_frame_age_sec)
        self.status_provider: Optional[Callable[[], dict]] = None
        self.gaps = WatchGapTracker(offline_queue)

        self._stop_worker = threading.Event()
        self._worker_thread = threading.Thread(target=self._retry_worker_loop, daemon=True)
        self._worker_thread.start()

    def _init_supabase(self) -> None:
        if cfg.supabase_url and cfg.supabase_key:
            try:
                self.supabase = create_client(cfg.supabase_url, cfg.supabase_key)
                log.info("Cliente Supabase inicializado correctamente.")
            except Exception as e:
                log.error(f"Fallo al conectar con Supabase SDK: {e}")
                self.supabase = None
        else:
            log.warning("SUPABASE_URL o SUPABASE_KEY ausentes. Modo estrictamente local/offline.")

    def stop(self) -> None:
        """Detiene el worker de reintentos."""
        self._stop_worker.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

    def save_and_dispatch(
        self,
        crops: List[np.ndarray],
        track_id: int,
        event_type: str,
        timestamp: Optional[int] = None,
        reason: str = "",
        plate_crops: Optional[List[np.ndarray]] = None,
        local_ocr: Optional[dict] = None,
        vehicle_class: str = "",
        crop_meta: Optional[dict] = None,
    ) -> None:
        """
        Guarda los recortes como WebP y encola el evento para despacho inmediato.
        El primer recorte es el principal (la foto del vehículo que ve el panel). Si hubo lectura
        local, le siguen las patentes ampliadas (lo que lee el OCR en la nube); si no, otros
        recortes del vehículo para que el OCR confirme la patente entre varios frames.
        """
        ts = timestamp or int(time.time())
        CAPTURES_DIR.mkdir(exist_ok=True)
        base = f"crop_vehicle_{track_id}_{ts}_{event_type.lower()}"

        named = [(f"{base}.webp" if i == 0 else f"{base}_f{i}.webp", c, 90) for i, c in enumerate(crops[:1])]
        # Patente: calidad alta, son pocos KB y cada borde de carácter cuenta
        named += [(f"{base}_plate{i}.webp", c, 97) for i, c in enumerate(plate_crops or [])]
        named += [(f"{base}_f{i}.webp", c, 90) for i, c in enumerate(crops[1:], start=1)]

        paths: List[str] = []
        for name, crop, quality in named:
            path = str(CAPTURES_DIR / name)
            if cv2.imwrite(path, crop, [int(cv2.IMWRITE_WEBP_QUALITY), quality]):
                paths.append(path)
            else:
                log.error(f"Error escribiendo crop en disco: {path}")

        if not paths:
            return

        event_id = self.offline_queue.enqueue(
            track_id=track_id,
            event_type=event_type,
            image_path=paths[0],
            timestamp=ts,
            camera_id=cfg.camera_id,
            sucursal_id=cfg.default_sucursal_id,
            extra_paths=paths[1:],
            reason=reason,
            local_ocr=local_ocr,
            vehicle_class=vehicle_class,
            crop_meta=crop_meta,
        )

        threading.Thread(target=self._process_single_event, args=(event_id,), daemon=True).start()

    def _process_single_event(self, event_id: str) -> bool:
        """Intenta subir y notificar un evento individual."""
        pending = self.offline_queue.get_pending(limit=500, due_only=False)
        target_event = next((ev for ev in pending if ev.id == event_id), None)
        if not target_event:
            return False
        return self._upload_and_notify(target_event)

    def _upload_and_notify(self, event: QueuedEvent) -> bool:
        with self._in_flight_lock:
            if event.id in self._in_flight:
                return False
            self._in_flight.add(event.id)
        try:
            return self._do_upload_and_notify(event)
        finally:
            with self._in_flight_lock:
                self._in_flight.discard(event.id)

    def _upload_file(self, local_path: str, sucursal_id: str) -> str:
        storage_path = f"{sucursal_id}/{os.path.basename(local_path)}"
        with open(local_path, "rb") as f:
            self.supabase.storage.from_(BUCKET).upload(
                storage_path, f,
                file_options={"content-type": "image/webp", "upsert": "true"},
            )
        return storage_path

    def _do_upload_and_notify(self, event: QueuedEvent) -> bool:
        """
        Sube los recortes a Supabase Storage y hace POST a WEBHOOK_URL.
        El evento solo se marca completado si la Edge Function confirma el proceso;
        así un corte de red o un error del webhook se reintenta (la Edge Function es
        idempotente por event_id).
        """
        max_age = cfg.offline_max_age_days * 86400
        if not self.supabase or not cfg.webhook_url:
            self.offline_queue.increment_attempts(event.id, max_age, "Supabase o WEBHOOK_URL sin configurar")
            return False

        if not os.path.exists(event.image_path):
            log.error(f"Archivo de imagen no encontrado: {event.image_path}. Marcando completado para evitar loop.")
            self.offline_queue.mark_completed(event.id, status="SIN_FOTO")
            return False

        try:
            storage_path = self._upload_file(event.image_path, event.sucursal_id)
            extra_storage = [
                self._upload_file(p, event.sucursal_id) for p in event.extra_paths if os.path.exists(p)
            ]
            log.info(f"Crop(s) subido(s) a Supabase Storage: {storage_path} (+{len(extra_storage)})")
        except Exception as e:
            log.warning(f"Error subiendo evento {event.id}: {e}")
            self.offline_queue.increment_attempts(event.id, max_age, f"storage: {e}")
            return False

        payload = {
            "source": "anpr-edge",
            "event_id": event.id,
            "filePath": storage_path,
            "extra_file_paths": extra_storage,
            "direction": event.event_type,          # ENTRADA | SALIDA | DESCONOCIDA
            "direction_reason": event.reason,
            "camera_id": event.camera_id,
            "sucursal_id": event.sucursal_id,
            "track_id": event.track_id,
            "timestamp": event.timestamp,
        }
        if event.vehicle_class:
            payload["vehicle_class"] = event.vehicle_class
        if event.crop_meta:
            # Calidad del recorte principal: la función elige con esto la foto de referencia de los abonados
            # y decide si el color de la foto sirve para compararlo con el registrado
            payload["crop_meta"] = event.crop_meta
        if event.local_ocr:
            # Segunda opinión independiente: la Edge Function acepta la patente sola solo si
            # coincide con Gemini (o si la lectura local es fuerte por sí misma).
            payload["local_ocr"] = event.local_ocr
        headers = self._webhook_headers()

        error = ""
        body: dict = {}
        try:
            resp = requests.post(cfg.webhook_url, json=payload, headers=headers, timeout=cfg.webhook_timeout_sec)
            try:
                body = resp.json()
            except ValueError:
                pass
            ok = resp.ok and not (isinstance(body, dict) and body.get("error"))
            log.info(f"Webhook HTTP {resp.status_code}: {resp.text[:300]}")
            if not ok:
                error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as w_err:
            log.warning(f"Webhook no respondió ({w_err}); se reintentará. La imagen ya está en Storage.")
            ok = False
            error = f"sin respuesta: {w_err}"

        if ok:
            self.offline_queue.mark_completed(event.id, body if isinstance(body, dict) else None)
            return True
        # Nunca se descarta por cantidad de intentos: se reintenta con espera creciente (hasta
        # 30 min) durante OFFLINE_MAX_AGE_DAYS. La función es idempotente y usa la hora del evento.
        self.offline_queue.increment_attempts(event.id, max_age, error)
        return False

    def _webhook_headers(self) -> dict:
        headers = {
            "Authorization": f"Bearer {cfg.supabase_key}",
            "apikey": cfg.supabase_key,
            "Content-Type": "application/json",
        }
        if cfg.agent_secret:
            # La Edge Function rechaza (401) llamadas sin este secreto cuando lo tiene configurado
            headers["x-agent-secret"] = cfg.agent_secret
        return headers

    def _report_status(self) -> None:
        """
        Latido al panel: la Edge Function guarda si el agente está vivo, si la cámara entrega
        video y cuántos eventos esperan en la cola. El panel avisa si deja de llegar.
        """
        if not cfg.webhook_url or cfg.agent_status_interval_sec <= 0:
            return
        now = time.monotonic()
        if self._last_status and now - self._last_status < cfg.agent_status_interval_sec:
            return
        self._last_status = now
        payload = {
            "type": "heartbeat",
            "sucursal_id": cfg.default_sucursal_id,
            "camera_id": cfg.camera_id,
            "queue_pending": self.offline_queue.count_pending(),
            "queue_failed": self.offline_queue.count_failed(),
            "version": AGENT_VERSION,
        }
        gaps = self.offline_queue.unsent_gaps()
        if gaps:
            payload["watch_gaps"] = [{k: g[k] for k in ("from", "to", "reason")} for g in gaps]
        if self.status_provider:
            try:
                payload.update(self.status_provider())
            except Exception as e:  # el latido nunca debe tumbar al worker
                log.debug(f"No se pudo leer el estado de la cámara: {e}")
        try:
            resp = requests.post(cfg.webhook_url, json=payload, headers=self._webhook_headers(), timeout=15.0)
            if not resp.ok:
                log.warning(f"Latido rechazado (HTTP {resp.status_code}): {resp.text[:200]}")
            elif gaps:
                self.offline_queue.mark_gaps_sent([g["id"] for g in gaps])
        except Exception as e:
            log.warning(f"Latido al panel falló: {e}")

    def _tick_gaps(self) -> None:
        """Anota si el portón estuvo sin vigilancia (PC suspendida/apagada o cámara sin imagen)."""
        status = {}
        if self.status_provider:
            try:
                status = self.status_provider() or {}
            except Exception:
                status = {}
        age = status.get("last_frame_age_sec")
        watching = bool(status.get("camera_ok")) and (age is None or age <= 60)
        try:
            self.gaps.tick(watching)
        except Exception as e:  # nunca debe tumbar al worker
            log.debug(f"Control de huecos falló: {e}")

    def _heartbeat(self) -> None:
        """
        Consulta mínima a la base: mantiene activo el proyecto Supabase (el plan
        gratuito lo pausa tras 7 días sin actividad; así se perdió la URL en 2026).
        """
        if not self.supabase or cfg.heartbeat_interval_sec <= 0:
            return
        now = time.monotonic()
        if self._last_heartbeat and now - self._last_heartbeat < cfg.heartbeat_interval_sec:
            return
        self._last_heartbeat = now
        try:
            self.supabase.table("Configuracion").select("id").limit(1).execute()
            log.info("💓 Heartbeat Supabase OK.")
        except Exception as e:
            log.warning(f"Heartbeat Supabase falló: {e}")

    def _retry_worker_loop(self) -> None:
        """Loop de fondo para reintentar eventos pendientes periódicamente."""
        log.info("Iniciando worker de reintentos offline...")
        while not self._stop_worker.is_set():
            try:
                self._heartbeat()
                self._tick_gaps()
                self._report_status()
                pending_events = self.offline_queue.get_pending(limit=5)
                if pending_events:
                    log.info(f"Worker offline: reintentando {len(pending_events)} eventos pendientes...")
                    for event in pending_events:
                        if self._stop_worker.is_set():
                            break
                        self._upload_and_notify(event)
                        time.sleep(1.0)
            except Exception as e:
                log.error(f"Error en worker loop offline: {e}")

            self._stop_worker.wait(timeout=cfg.offline_retry_interval_sec)
