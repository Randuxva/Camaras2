"""
Reproduce una grabación de tools/record_streams.py a través del pipeline real (sin red ni nube)
en tiempo real, como si fuera la cámara, para comparar versiones con los mismos autos.

Uso (desde anpr-edge/):
  venv/Scripts/python.exe tools/replay.py <dir_grabacion> <salida> [--single] [--motion-heatmap]

  --single          modo anterior: un solo stream (principal) para detección y recortes
  --motion-heatmap  además guarda dónde hay movimiento (para definir máscaras)

Salida: events.jsonl (evento, lectura local, tiempos), crops/ (lo que se enviaría), frames/
(cuadros anotados cada vez que hay un vehículo cerca del portón) y metrics.jsonl (CPU, fps).
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path

import av
import cv2
import numpy as np
import psutil

AGENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT))

from config import cfg  # noqa: E402
from core.frame_selector import FrameBuffer  # noqa: E402
from core.image_quality import ANPRImageEnhancer  # noqa: E402
from core.plate_reader import PlateReader  # noqa: E402
from core.quietos import RegistroQuietos  # noqa: E402
from core.repetidos import RegistroRepetidos  # noqa: E402
from core.roi_engine import ROIEngine  # noqa: E402
from core.tracker_state import TrackerState  # noqa: E402
from pipeline.consumer import InferenceConsumer  # noqa: E402


def first_arrival(path: Path):
    """(pts, hora de llegada) del primer paquete grabado: alinea los dos streams como llegaron."""
    d = json.loads(path.with_suffix(".jsonl").open().readline())
    return d["pts"], d["wall"]


def due_fn(path: Path, t0_mono: float, ref_wall: float):
    pts0, wall0 = first_arrival(path)
    def due(frame_or_packet) -> float:
        return t0_mono + (wall0 - ref_wall) + float((frame_or_packet.pts - pts0) * frame_or_packet.time_base)
    return due


class FileMain:
    """
    Imita MainStreamBuffer leyendo main.mkv al ritmo del reloj: mismos modos (auto/all/keyframes),
    misma prioridad baja y misma publicación (hora de llegada, cuadro) para el consumidor.
    """

    def __init__(self, path: Path, due, mode: str):
        self.path, self.due, self.mode = path, due, mode
        self._new: list = []
        self.lock = threading.Lock()
        self.shape = None
        self.last_frame_monotonic = 0.0
        self.decoded = 0
        self.stop = False
        self.switched_at = None

    def pop_new(self):
        with self.lock:
            out, self._new = self._new, []
        return out

    def run(self):
        from collections import deque
        from utils.priority import lower_current_thread_priority
        lower_current_thread_priority()
        c = av.open(str(self.path))
        s = c.streams.video[0]
        fps = float(s.average_rate or 15)
        keyframes_only = self.mode == "keyframes"
        if keyframes_only:
            s.codec_context.skip_frame = "NONKEY"
            s.codec_context.thread_count = 1
        else:
            s.thread_type = "AUTO"
        times: deque = deque(maxlen=60)
        for pkt in c.demux(s):
            if self.stop:
                break
            if pkt.pts is None:
                continue
            due = self.due(pkt)
            delay = due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            t = time.perf_counter()
            for f in pkt.decode():
                with self.lock:
                    self._new.append((self.due(f), f))   # por el PTS del cuadro, como el productor
                    self._new = self._new[-6:]
                self.shape = (f.height, f.width)
                self.last_frame_monotonic = time.monotonic()
                self.decoded += 1
            if self.mode == "auto" and not keyframes_only:
                times.append(time.perf_counter() - t)
                if len(times) == times.maxlen and sum(times) / len(times) * fps > 0.9:
                    print(f"auto: el equipo no llega con todos los cuadros; pasa a keyframes ({time.strftime('%H:%M:%S')})")
                    self.mode, keyframes_only = "keyframes", True
                    self.switched_at = time.time()
                    s.codec_context.skip_frame = "NONKEY"
        c.close()


class FakeUploader:
    def __init__(self, out: Path):
        self.out = out
        (out / "crops").mkdir(parents=True, exist_ok=True)

    def save_and_dispatch(self, crops, track_id, event_type, timestamp=None, reason="", **kw):
        stamp = time.strftime("%H%M%S")
        names = []
        for kind, imgs in (("car", crops), ("plate", kw.get("plate_crops") or [])):
            for i, c in enumerate(imgs):
                n = f"{stamp}_t{track_id}_{event_type.lower()}_{kind}{i}.jpg"
                cv2.imwrite(str(self.out / "crops" / n), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
                names.append(n)
        with open(self.out / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"wall": time.time(), "tid": track_id, "event": event_type, "reason": reason,
                                "local_ocr": kw.get("local_ocr"), "crops": names}, ensure_ascii=False) + "\n")


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    single = "--single" in sys.argv
    heat = "--motion-heatmap" in sys.argv
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)

    det_path = src / ("main.mkv" if single else "sub.mkv")
    q: queue.Queue = queue.Queue(maxsize=1)
    t0_mono = time.monotonic() + 3.0
    ref_wall = first_arrival(det_path)[1]
    det_due = due_fn(det_path, t0_mono, ref_wall)

    main_src = None
    if not single:
        main_src = FileMain(src / "main.mkv", due_fn(src / "main.mkv", t0_mono, ref_wall), cfg.main_decode_mode)
        threading.Thread(target=main_src.run, daemon=True).start()

    uploader = FakeUploader(out)
    roi = ROIEngine(line_a_y_ratio=cfg.roi_line_a_y_ratio, line_b_y_ratio=cfg.roi_line_b_y_ratio,
                    direction_timeout_sec=cfg.roi_direction_timeout_sec, polygon_ratios=cfg.roi_polygon_ratios,
                    track_lost_sec=cfg.roi_track_lost_sec, min_displacement_ratio=cfg.roi_min_displacement_ratio,
                    continuation_sec=cfg.roi_continuation_sec, event_hold_sec=cfg.roi_event_hold_sec,
                    solo_movimiento=cfg.roi_solo_movimiento in ("true", "1", "yes", "sombra"),
                    solo_cruces=cfg.roi_solo_cruces in ("true", "1", "yes", "sombra"), union_sec=cfg.roi_union_sec)
    estricta = cfg.roi_solo_cruces in ("true", "1", "yes", "sombra")
    consumer = InferenceConsumer(
        frame_queue=q, uploader=uploader,
        tracker_state=TrackerState(cfg.max_processed_ids_ttl, cfg.cooldown_per_plate_sec, cfg.max_active_tracks),
        roi_engine=roi, frame_buffer=FrameBuffer(cfg.frame_buffer_size, cfg.min_blur_threshold),
        enhancer=ANPRImageEnhancer(),
        plate_reader=PlateReader.try_create(cfg.plate_detector_model, cfg.plate_ocr_model, threads=cfg.plate_reader_threads),
        main_source=main_src,
        quietos=(RegistroQuietos(str(out / "quietos.db"), str(out / "quietos"),
                                 cfg.roi_solo_cruces if estricta else cfg.roi_solo_movimiento, estricto=estricta)
                 if estricta or cfg.roi_solo_movimiento in ("true", "1", "yes", "sombra") else None),
        repetidos=(RegistroRepetidos(str(out / "quietos.db"), str(out / "repetidos"), cfg.roi_agrupar_repetidos)
                   if cfg.roi_agrupar_repetidos in ("true", "1", "yes", "sombra") else None),
    )

    # Registrar cada inferencia y guardar cuadros anotados con vehículos cerca del portón
    from ultralytics import YOLO
    real_track = YOLO.track
    last_snap = [0.0]

    def track(self, frame, *a, **k):
        t = time.perf_counter()
        r = real_track(self, frame, *a, **k)
        ms = (time.perf_counter() - t) * 1000
        b = r[0].boxes
        boxes = b.xyxy.int().cpu().tolist() if b is not None and len(b) else []
        ids = b.id.int().cpu().tolist() if b is not None and b.id is not None else [None] * len(boxes)
        h, w = frame.shape[:2]
        near = [bx for bx in boxes if bx[3] / h >= cfg.roi_line_b_y_ratio - 0.1]
        snap = None
        if near and time.monotonic() - last_snap[0] > 0.4:
            last_snap[0] = time.monotonic()
            img = cv2.resize(frame, (960, int(h * 960 / w)))
            s = 960 / w
            for y, col in ((cfg.roi_line_b_y_ratio, (0, 255, 255)), (cfg.roi_line_a_y_ratio, (0, 0, 255))):
                cv2.line(img, (0, int(y * h * s)), (960, int(y * h * s)), col, 1)
            for bx, i in zip(boxes, ids):
                cv2.rectangle(img, (int(bx[0] * s), int(bx[1] * s)), (int(bx[2] * s), int(bx[3] * s)), (0, 255, 0), 2)
                cv2.putText(img, f"#{i}", (int(bx[0] * s), int(bx[1] * s) - 4), 0, 0.5, (0, 255, 0), 1)
            snap = time.strftime("%H%M%S") + f"{time.time() % 1:.1f}"[1:] + ".jpg"
            cv2.imwrite(str(out / "frames" / snap), img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with open(out / "infer.jsonl", "a") as f:
            f.write(json.dumps({"wall": time.time(), "ms": round(ms, 1), "n": len(boxes), "ids": ids,
                                "yb": [round(bx[3] / h, 3) for bx in boxes], "snap": snap}) + "\n")
        return r
    YOLO.track = track

    heat_acc = None
    heat_mog = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
    stop = threading.Event()

    def metrics():
        p = psutil.Process()
        p.cpu_percent()
        while not stop.is_set():
            time.sleep(5)
            with open(out / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"wall": time.time(), "cpu_proc": p.cpu_percent(), "rss_mb": p.memory_info().rss // 10**6,
                                    "infer": consumer.inference_counter}) + "\n")
    threading.Thread(target=metrics, daemon=True).start()

    consumer.start()
    c = av.open(str(det_path))
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    n = 0
    for f in c.decode(s):
        due = det_due(f)
        delay = due - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        img = f.to_ndarray(format="bgr24")
        if heat:
            mask = heat_mog.apply(cv2.resize(img, (160, 120), interpolation=cv2.INTER_NEAREST))
            heat_acc = (mask > 0).astype(np.float32) if heat_acc is None else heat_acc + (mask > 0)
        if q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        q.put_nowait((img, time.time(), time.monotonic()))
        n += 1
    c.close()
    time.sleep(8)  # dejar cerrar los tracks pendientes
    consumer.stop()
    stop.set()
    if main_src:
        main_src.stop = True
    if heat_acc is not None:
        hm = cv2.applyColorMap(cv2.normalize(heat_acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
        np.save(out / "motion_heat.npy", heat_acc)
        cv2.imwrite(str(out / "motion_heat.jpg"), cv2.resize(hm, (640, 480), interpolation=cv2.INTER_NEAREST))
    print(f"ok: {n} cuadros de detección, {consumer.inference_counter} inferencias, "
          f"principal: {main_src.decoded if main_src else 0} cuadros decodificados, modo final {main_src.mode if main_src else '-'}")


if __name__ == "__main__":
    main()
