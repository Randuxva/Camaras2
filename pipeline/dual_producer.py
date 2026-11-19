"""
Captura con dos streams (práctica estándar de los NVR, p. ej. Frigate: "detect" chico + principal).

POR QUÉ (medido en el Pentium B980, 2026-10-02):
  - OpenCV leyendo el principal 2880x1620 H.265 a 15 fps: ~60 % de CPU (de 200 %) solo en
    decodificar y convertir CADA cuadro a BGR, aunque YOLO usaba ~1 de cada 10.
  - Sub-stream 640x360: ~2 %. YOLO a imgsz=320 tarda lo mismo con cualquiera de los dos.

Diseño:
  - Sub-stream (OpenCV) → cola de detección (drop-oldest), igual que FrameProducer.
  - Principal (PyAV) en su propio hilo de prioridad baja: publica cada cuadro decodificado con su
    hora de llegada; solo se convierte a BGR si hay un vehículo en el portón en ese instante.
  - El principal decodifica solo keyframes (MAIN_DECODE_MODE=keyframes, por defecto): ver
    MainStreamBuffer.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

from camera import open_ip_source, _safe_source_label
from config import cfg
from utils.logger import get_logger
from utils.priority import lower_current_thread_priority

log = get_logger(__name__)


class MainStreamBuffer:
    """
    Decodifica el stream principal y publica sus cuadros (con la hora de llegada del paquete) para
    que el consumidor recorte los vehículos con la posición que tenían en el sub-stream.

    Modos (MAIN_DECODE_MODE):
      "auto" (por defecto): decodifica todos los cuadros; si el hardware no llega a tiempo real
             (decodificar un cuadro tarda más que el intervalo entre cuadros, promedio de 60),
             pasa solo a keyframes y lo registra. En una mini PC queda en "all". Se mide el tiempo
             de decodificación y no el atraso, para que un corte de red no lo active.
      "all": todos los cuadros (más vistas de cada patente; necesita hardware que lo sostenga).
      "keyframes": solo keyframes (1 cada 2 s en esta cámara).
    Medido el 2026-10-02 en el Pentium B980 (equipo de prueba): todos los cuadros de 2880x1620 H.265
    no llegan a tiempo real (8,5–10,9 fps de 15); solo keyframes cuesta ~6 % de un núcleo, y son los
    cuadros de mayor calidad (las 3 patentes de autos de la grabación, confianza 0,93–1,00).
    Con la cámara en "I-frame interval = 15" (DMSS) hay un keyframe por segundo: el doble de vistas.
    """

    LOAD_LIMIT = 0.9    # en "auto": fracción del intervalo entre cuadros que puede ocupar decodificar

    def __init__(self, url: str, mode: str = "auto", keep: int = 6) -> None:
        self.url = url
        self.mode = mode
        self._new: deque = deque(maxlen=keep)   # (monotonic de llegada, av.VideoFrame) sin consumir
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_frame_monotonic = 0.0
        self.shape: Optional[Tuple[int, int]] = None
        self.decoded = 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def pop_new(self) -> List[Tuple[float, "object"]]:
        """Cuadros decodificados desde la última llamada: [(monotonic de llegada, av.VideoFrame)]."""
        with self._lock:
            out = list(self._new)
            self._new.clear()
        return out

    def _publish(self, arrival: float, frame) -> None:
        with self._lock:
            self._new.append((arrival, frame))
        self.last_frame_monotonic = time.monotonic()
        self.shape = (frame.height, frame.width)
        self.decoded += 1

    def _loop(self) -> None:
        import av

        # Decodifica en este hilo con prioridad baja: la detección (YOLO) va primero
        lower_current_thread_priority()
        while not self._stop.is_set():
            try:
                container = av.open(self.url, options={"rtsp_transport": "tcp", "fflags": "nobuffer"},
                                    timeout=cfg.camera_open_timeout_ms / 1000)
            except Exception as e:
                log.warning(f"Stream principal no abrió ({_safe_source_label(self.url)}): {e}")
                self._stop.wait(cfg.reconnect_delay_sec)
                continue
            stream = container.streams.video[0]
            ctx = stream.codec_context
            keyframes_only = self.mode == "keyframes"
            if keyframes_only:
                ctx.skip_frame = "NONKEY"
                ctx.thread_count = 1          # un keyframe cada 1–2 s: un hilo (de prioridad baja) alcanza
            else:
                stream.thread_type = "AUTO"   # todos los núcleos que haga falta (mini PC)
            log.info(f"Stream principal abierto: {ctx.width}x{ctx.height} {ctx.name} "
                     f"({'solo keyframes' if keyframes_only else 'todos los cuadros'}, modo {self.mode})")
            decode_times: deque = deque(maxlen=60)
            # El decodificador puede entregar un cuadro recién al recibir el paquete siguiente (en modo
            # keyframes, 2 s después): cada cuadro se marca con la llegada de SU paquete (por PTS).
            # Sin esto los recortes salían del momento equivocado (auto todavía en la calle).
            arrival_by_pts: dict = {}
            fps = float(stream.average_rate or 15)
            try:
                for packet in container.demux(stream):
                    if self._stop.is_set():
                        break
                    # En modo keyframes los demás paquetes se entregan igual: el decodificador los
                    # descarta sin decodificarlos (skip_frame=NONKEY), pero sin ellos retenía cada
                    # keyframe hasta el siguiente (2 s tarde).
                    arrival = time.monotonic()   # hora de llegada del paquete (no de fin de decodificación)
                    if packet.pts is not None:
                        arrival_by_pts[packet.pts] = arrival
                        if len(arrival_by_pts) > 120:
                            for k in sorted(arrival_by_pts)[:60]:
                                del arrival_by_pts[k]
                    t_dec = time.perf_counter()
                    for frame in packet.decode():
                        self._publish(arrival_by_pts.pop(frame.pts, arrival), frame)
                    if self.mode == "auto" and not keyframes_only:
                        decode_times.append(time.perf_counter() - t_dec)
                        load = sum(decode_times) / len(decode_times) * fps
                        if len(decode_times) == decode_times.maxlen and load > self.LOAD_LIMIT:
                            log.warning(f"El equipo no decodifica el stream principal en tiempo real "
                                        f"(usa {load:.0%} del tiempo disponible): se usan solo keyframes.")
                            # En caliente, sin reconectar: reabrir dejaba ~2 s sin imagen y en la
                            # prueba el cambio ocurrió justo mientras salía un auto.
                            self.mode = "keyframes"
                            keyframes_only = True
                            ctx.skip_frame = "NONKEY"
            except Exception as e:
                log.warning(f"Stream principal interrumpido: {e}")
            finally:
                try:
                    container.close()
                except Exception:
                    pass
            if not self._stop.is_set():
                self._stop.wait(cfg.reconnect_delay_sec)   # corte de red: esperar antes de reconectar


class DualStreamProducer:
    """Sub-stream para detección (cola) + stream principal bajo demanda para los recortes."""

    def __init__(self, frame_queue: queue.Queue, detect_url: str, main_url: str) -> None:
        self.frame_queue = frame_queue
        self.detect_url = detect_url
        self.main = MainStreamBuffer(main_url, mode=cfg.main_decode_mode)
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_frame_monotonic = 0.0
        self.read_failures = 0
        self.camera_label = _safe_source_label(detect_url)

    def start(self) -> bool:
        self.cap = open_ip_source(self.detect_url)
        if not self.cap:
            log.error(f"No se pudo abrir el sub-stream de detección: {self.camera_label}")
            return False
        log.info(f"Sub-stream de detección abierto: {self.camera_label}")
        self.main.start()
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.main.stop()
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        log.info("Recursos de captura (dos streams) liberados.")

    def status(self) -> dict:
        now = time.monotonic()
        ages = [now - t for t in (self.last_frame_monotonic, self.main.last_frame_monotonic) if t]
        age = max(ages) if len(ages) == 2 else None
        return {
            # Sin el principal no hay foto de patente: la cámara cuenta como caída
            "camera_ok": bool(age is not None and age < 30),
            "last_frame_age_sec": round(age, 1) if age is not None else None,
        }

    def _capture_loop(self) -> None:
        while not self.stop_event.is_set():
            if not self.cap or not self.cap.isOpened():
                time.sleep(cfg.reconnect_delay_sec)
                self.cap = open_ip_source(self.detect_url)
                continue
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.read_failures += 1
                if self.read_failures >= cfg.ip_read_failures_before_reconnect:
                    log.warning("Sub-stream sin cuadros: reconectando...")
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                    self.read_failures = 0
                else:
                    time.sleep(0.01)
                continue
            self.read_failures = 0
            now = time.monotonic()
            self.last_frame_monotonic = now
            try:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait((frame, time.time(), now))
            except Exception as e:
                log.error(f"Error encolando cuadro de detección: {e}")
