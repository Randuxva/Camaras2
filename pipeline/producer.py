"""
Producer Thread para captura de video continua y sin lag (2026).
Lee continuamente desde OpenCV y coloca los frames en una cola de tamaño 1 (Drop-Oldest).
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import cv2

from camera import open_camera
from config import cfg
from utils.logger import get_logger

log = get_logger(__name__)


class FrameProducer:
    """
    Lee continuamente de la fuente de video en un hilo dedicado.
    Evita la acumulación de latencia en buffers RTSP descartando frames antiguos.
    """

    def __init__(self, frame_queue: queue.Queue) -> None:
        self.frame_queue = frame_queue
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_label: str = ""
        self.is_ip_camera: bool = False

        self.read_failures = 0
        self.reconnect_attempts = 0
        self.last_reconnect_time = time.time()
        # Último frame leído OK (monotonic): el latido al panel lo usa para avisar "cámara sin video"
        self.last_frame_monotonic: float = 0.0

    def start(self) -> bool:
        """Inicia la cámara y el hilo de captura."""
        self.cap, self.camera_label, self.is_ip_camera = open_camera()
        if not self.cap or not self.cap.isOpened():
            log.error(f"No se pudo abrir la cámara inicial: {self.camera_label}")
            return False

        log.info(f"Cámara abierta exitosamente: {self.camera_label} (IP: {self.is_ip_camera})")
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def status(self) -> dict:
        """Estado de la cámara para el latido: conectada y segundos desde el último frame."""
        age = time.monotonic() - self.last_frame_monotonic if self.last_frame_monotonic else None
        return {
            "camera_ok": bool(self.cap is not None and self.cap.isOpened() and age is not None and age < 30),
            "last_frame_age_sec": round(age, 1) if age is not None else None,
        }

    def stop(self) -> None:
        """Detiene el hilo y libera los recursos de la cámara."""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        log.info("Recursos del productor de cámara liberados.")

    def _reconnect(self) -> bool:
        log.warning(f"Reconectando stream de cámara: {self.camera_label}...")
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        time.sleep(cfg.reconnect_delay_sec)
        self.cap, self.camera_label, self.is_ip_camera = open_camera()
        self.last_reconnect_time = time.time()
        self.read_failures = 0
        if self.cap and self.cap.isOpened():
            log.info(f"Reconexión exitosa: {self.camera_label}")
            return True
        log.error("Fallo al reconectar la cámara.")
        return False

    def _capture_loop(self) -> None:
        log.info("Iniciando bucle de captura en productor de video...")
        while not self.stop_event.is_set():
            # Reconexión proactiva periódica (opcional; deja ~5 s ciegos en cada ciclo, 0 = desactivada)
            if (
                self.is_ip_camera
                and cfg.proactive_reconnect_interval_sec > 0
                and time.time() - self.last_reconnect_time >= cfg.proactive_reconnect_interval_sec
            ):
                log.info(f"Ejecutando reconexión proactiva ({cfg.proactive_reconnect_interval_sec}s)...")
                self._reconnect()
                continue

            if not self.cap or not self.cap.isOpened():
                if not self._reconnect():
                    time.sleep(2.0)
                continue

            success, frame = self.cap.read()

            if not success or frame is None:
                self.read_failures += 1
                if self.read_failures >= cfg.ip_read_failures_before_reconnect:
                    self.reconnect_attempts += 1
                    if self.reconnect_attempts > cfg.max_reconnects_before_exit:
                        log.critical("Límite de reconexiones alcanzado en productor. Deteniendo...")
                        break
                    self._reconnect()
                else:
                    time.sleep(0.01)
                continue

            self.read_failures = 0
            self.reconnect_attempts = 0  # el límite es de reconexiones CONSECUTIVAS, no acumuladas
            self.last_frame_monotonic = time.monotonic()

            # Estrategia Drop-Oldest: si la cola está llena, descartar el frame viejo y meter el nuevo
            try:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                # (frame, epoch para registros/nombres de archivo, monotónico para lógica de tiempos)
                self.frame_queue.put_nowait((frame, time.time(), time.monotonic()))
            except Exception as e:
                log.error(f"Error encolando frame: {e}")

        log.info("Bucle de captura terminado.")
