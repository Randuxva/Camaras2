"""
Consumer Thread para procesamiento de IA, tracking y despacho de eventos (2026).
Integra:
  - YOLO / BoT-SORT con clases de vehículos
  - Prefiltrado de movimiento MOG2 para ahorro de CPU en reposo
  - ROI Engine por trayectoria (ENTRADA / SALIDA / DESCONOCIDA)
  - Buffer de mejores recortes y mejora óptica (CLAHE / Gamma / Denoise)
  - Despacho de los N mejores recortes hacia CloudUploader (lectura multi-frame)
"""
from __future__ import annotations

import queue
from collections import Counter, deque
import threading
import time
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from cloud.uploader import CloudUploader
from config import cfg
from core.frame_selector import FrameBuffer
from core.image_quality import ANPRImageEnhancer, analyze_lighting
from core.plate_reader import PlateReader, fuse_reads, upscale_plate
from utils.priority import lower_current_thread_priority
from core.roi_engine import ROIEngine, ROIEvent
from core.vehicle_attrs import atributos_de
from core.quietos import ARCHIVAR_SIN_CRUCE, SUPRIMIR, RegistroQuietos, realzar
from core.repetidos import AGRUPAR, RegistroRepetidos
from core.tracker_state import TrackerState
from utils.logger import get_logger

log = get_logger(__name__)

# Clases COCO que sigue el detector (cfg.vehicle_classes)
COCO_VEHICULOS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class InferenceConsumer:
    """
    Consume frames de la cola, ejecuta inferencia YOLO con seguimiento de IDs,
    evalúa trayectorias en el ROI Engine y despacha los mejores recortes.
    """

    def __init__(
        self,
        frame_queue: queue.Queue,
        uploader: CloudUploader,
        tracker_state: TrackerState,
        roi_engine: ROIEngine,
        frame_buffer: FrameBuffer,
        enhancer: ANPRImageEnhancer,
        plate_reader: Optional[PlateReader] = None,
        main_source=None,
        quietos: Optional[RegistroQuietos] = None,
        repetidos: Optional[RegistroRepetidos] = None,
    ) -> None:
        self.frame_queue = frame_queue
        self.uploader = uploader
        self.tracker_state = tracker_state
        self.roi_engine = roi_engine
        self.frame_buffer = frame_buffer
        self.enhancer = enhancer
        self.plate_reader = plate_reader
        self.repetidos = repetidos        # agrupa los avisos repetidos de un mismo vehículo (None = apagado)
        self.repetidos_agrupados = 0
        self.quietos = quietos            # salvaguardas de los objetos quietos (None = sin ROI_SOLO_MOVIMIENTO)
        # Con dos streams: YOLO corre sobre el sub-stream y los recortes salen del principal
        self.main_source = main_source
        self._last_main_crop: dict = {}
        self._bbox_hist: dict = {}       # track_id -> deque[(monotonic, bbox sub-stream, (h, w))]
        self._track_cls: dict = {}   # track_id → votos por clase del detector
        self._sub_fallback: dict = {}    # track_id -> último recorte del sub-stream (respaldo)
        self._last_infer_mono = 0.0
        self.quietos_descartados = 0     # objetos quietos sin patente que no se enviaron (ROI_SOLO_MOVIMIENTO)

        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        # La lectura de patente (~0,3 s por recorte en el Pentium) corre aparte para no frenar el video
        self._plate_jobs: "queue.Queue" = queue.Queue(maxsize=20)
        self._plate_thread: Optional[threading.Thread] = None

        self.model: Optional[YOLO] = None
        self.mog2 = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)

        self.last_annotated_frame: Optional[np.ndarray] = None
        self.frame_counter = 0
        self.inference_counter = 0
        self.last_inference_ms = 0.0

    def start(self) -> bool:
        """Carga el modelo YOLO e inicia el hilo de consumo."""
        log.info(f"Cargando modelo YOLO: {cfg.model_name}...")
        try:
            self.model = YOLO(cfg.model_name)
            log.info("Modelo YOLO cargado exitosamente.")
        except Exception as e:
            log.error(f"Error cargando modelo YOLO ({cfg.model_name}): {e}")
            return False

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()
        if self.plate_reader:
            self._plate_thread = threading.Thread(target=self._plate_loop, daemon=True)
            self._plate_thread.start()
        return True

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self._plate_thread and self._plate_thread.is_alive():
            # termina de enviar los eventos ya decididos (cada uno tarda pocos segundos)
            self._plate_thread.join(timeout=15.0)
        log.info("Consumidor de inferencia detenido.")

    def _check_motion(self, frame: np.ndarray) -> bool:
        """
        Calcula el porcentaje de píxeles en movimiento en escala reducida.
        Retorna True si hay suficiente movimiento para justificar inferencia YOLO.
        """
        if not cfg.use_mog2_prefilter:
            return True

        small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_NEAREST)
        fg_mask = self.mog2.apply(small)
        motion_ratio = float(np.count_nonzero(fg_mask)) / fg_mask.size
        return motion_ratio >= cfg.mog2_motion_threshold

    def _vehicle_class(self, track_id: int) -> str:
        """Clase que más veces vio el detector para este vehículo (car, motorcycle, bus, truck)."""
        votos = self._track_cls.get(track_id)
        if not votos:
            return ""
        return COCO_VEHICULOS.get(votos.most_common(1)[0][0], "")

    @staticmethod
    def _meta_de(c) -> dict:
        """Calidad de un recorte candidato, para que la nube elija la foto de referencia de un abonado y sepa si su color sirve."""
        hsv = cv2.cvtColor(cv2.resize(c.crop, (96, 72), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        return {
            "score": round(float(c.score), 1),
            "rejilla": bool(getattr(c, "rejilla", False)),
            "truncated": bool(c.truncated),
            "lighting": c.lighting.condition if c.lighting else None,
            # Infrarrojo o casi sin color: la foto no sirve para decir de qué color es el auto
            "sinColor": bool(np.percentile(hsv[..., 1], 90) < 25),
            "ancho": int(c.crop.shape[1]),
            "alto": int(c.crop.shape[0]),
        }

    def _con_atributos(self, meta: Optional[dict], crop: np.ndarray, track_id: int) -> Optional[dict]:
        """Suma color y tipo medidos aquí (sin IA) a la calidad del recorte: respaldo para cuando Gemini no responde."""
        try:
            return {**(meta or {}), **atributos_de(crop, self._vehicle_class(track_id))}
        except Exception as e:  # el respaldo nunca debe frenar el envío
            log.warning(f"No se pudo medir color/tipo local: {e}", extra={"track_id": track_id})
            return meta

    def _dispatch(self, event: ROIEvent, fallback_crop: Optional[np.ndarray], wall_ts: float) -> None:
        """Mejora y envía los mejores recortes del track para este evento."""
        track_id = event.track_id
        if self.tracker_state.is_track_processed(track_id):
            self.frame_buffer.clear_track(track_id)
            return
        self.tracker_state.mark_track_processed(track_id)

        if self.plate_reader:
            # Todos los candidatos del buffer, sin mejorar ni comprimir: el detector de patente
            # elige la mejor vista. Si la cola está llena se cae al envío clásico.
            cands = self.frame_buffer.get_top(track_id, cfg.frame_buffer_size)
            raw = [c.crop for c in cands]
            metas = [self._meta_de(c) for c in cands]
            if not raw and fallback_crop is None:
                fallback_crop = self._sub_fallback.get(track_id)
            if not raw and fallback_crop is not None and fallback_crop.size > 0:
                raw = [fallback_crop.copy()]
            if raw:
                try:
                    self._plate_jobs.put_nowait((event, raw, wall_ts, metas))
                    self.frame_buffer.clear_track(track_id)
                    return
                except queue.Full:
                    log.warning("Cola de lectura de patentes llena: envío sin lectura local.")

        candidates = self.frame_buffer.get_top(track_id, cfg.crops_per_event)
        crops: List[np.ndarray] = [c.crop for c in candidates]
        crop_meta = self._meta_de(candidates[0]) if candidates else None
        if not crops and fallback_crop is not None and fallback_crop.size > 0:
            crops = [fallback_crop]
        self.frame_buffer.clear_track(track_id)
        if not crops:
            log.warning("Evento sin recortes válidos, se descarta.", extra={"track_id": track_id})
            return

        enhanced = [self.enhancer.enhance(c)[0] for c in crops]
        crop_meta = self._con_atributos(crop_meta, enhanced[0], track_id)
        log.info(
            f"📸 Vehículo {event.event_type.value} (track #{track_id}, {event.reason}). "
            f"Enviando {len(enhanced)} recorte(s)..."
        )
        self.uploader.save_and_dispatch(
            crops=enhanced,
            track_id=track_id,
            event_type=event.event_type.value,
            timestamp=int(wall_ts),
            reason=event.reason,
            vehicle_class=self._vehicle_class(track_id),
            crop_meta=crop_meta,
        )

    def _plate_loop(self) -> None:
        """Lee la patente de cada evento y lo despacha (hilo propio, prioridad baja)."""
        lower_current_thread_priority()
        while not (self.stop_event.is_set() and self._plate_jobs.empty()):
            try:
                event, raw, wall_ts, metas = self._plate_jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._dispatch_with_plates(event, raw, wall_ts, metas)
            except Exception as e:  # nunca perder el evento por un fallo de la lectura local
                log.error(f"Lectura local falló ({e}); envío sin ella.", extra={"track_id": event.track_id})
                crops = [self.enhancer.enhance(c)[0] for c in raw[:cfg.crops_per_event]]
                meta = self._con_atributos(metas[0] if metas else None, crops[0], event.track_id) if crops else None
                self.uploader.save_and_dispatch(crops=crops, track_id=event.track_id,
                                                event_type=event.event_type.value,
                                                timestamp=int(wall_ts), reason=event.reason,
                                                vehicle_class=self._vehicle_class(event.track_id),
                                                crop_meta=meta)

    def _dispatch_with_plates(self, event: ROIEvent, raw: List[np.ndarray], wall_ts: float, metas: Optional[list] = None) -> None:
        t0 = time.perf_counter()
        reads = self.plate_reader.read_many(raw)
        fused = fuse_reads(reads)
        quieto = bool(event.sin_movimiento and self.quietos is not None)
        segunda = False
        if quieto and not fused.plate:
            # Segunda oportunidad antes de decidir: la misma imagen ampliada y con el contraste mejorado
            try:
                extra = self.plate_reader.read_many([realzar(c) for c in raw])
                if extra:
                    reads = list(reads) + list(extra)
                    fused = fuse_reads(reads)
                    segunda = bool(fused.plate)
            except Exception as e:  # la segunda lectura nunca debe frenar el evento
                log.warning(f"Segunda lectura del quieto falló ({e}).", extra={"track_id": event.track_id})
        if quieto:
            clase = self._vehicle_class(event.track_id)
            decision, rep = self.quietos.decidir(sin_movimiento=True, con_patente=bool(fused.plate),
                                                 clase=clase, caja=event.norm_box)
            self.quietos.registrar(track_id=event.track_id, tipo=event.event_type.value, ruta=event.reason,
                                   n_puntos=event.n_points, caja=event.norm_box, clase=clase,
                                   con_patente=bool(fused.plate), segunda_lectura=segunda, repeticiones=rep,
                                   decision=decision, crops=raw[:1])
            if decision in (SUPRIMIR, ARCHIVAR_SIN_CRUCE):
                # No es un paso por el portón: no va a revisión (regla estricta: no cruzó la línea; o lugar
                # recurrente, sin patente, no es moto).
                # La constancia y la foto quedan guardadas para siempre en quietos.db y la carpeta quietos/.
                self.quietos_descartados += 1
                log.debug(f"Objeto quieto en lugar recurrente (track #{event.track_id}, {event.reason}): no se envía.",
                          extra={"track_id": event.track_id, "n_points": event.n_points})
                return
        if self.repetidos is not None:
            firme = bool(fused.plate and fused.is_strong())
            direccion = event.event_type.value
            cola = getattr(self.uploader, "offline_queue", None)
            decision_r, desde = self.repetidos.decidir(
                patente=fused.plate, firme=firme, direccion=direccion,
                resultado_previo=getattr(cola, "accion_previa", None))
            self.repetidos.registrar(patente=fused.plate, firme=firme, direccion=direccion, track_id=event.track_id,
                                     ruta=event.reason, decision=decision_r, desde_anterior=desde, crops=raw[:1])
            if decision_r == AGRUPAR:
                # Mismo vehículo, misma patente firme y mismo sentido hace menos de 10 min: ya hay un aviso.
                # La constancia y la foto quedan guardadas para siempre en quietos.db y la carpeta repetidos/.
                self.repetidos_agrupados += 1
                log.debug(f"Aviso repetido del mismo vehículo (track #{event.track_id}, {direccion}): no se envía.",
                          extra={"track_id": event.track_id})
                return
        views = sorted(reads, key=lambda r: r.quality, reverse=True)[:cfg.plate_crops_per_event]

        metas = metas or []
        if views:
            # La foto del vehículo es la del cuadro donde mejor se ve la patente
            crops = [self.enhancer.enhance(raw[views[0].source_index])[0]]
            crop_meta = metas[views[0].source_index] if views[0].source_index < len(metas) else None
            plate_crops = [upscale_plate(r.plate_img) for r in views]
        else:
            crops = [self.enhancer.enhance(c)[0] for c in raw[:cfg.crops_per_event]]
            plate_crops = None
            crop_meta = metas[0] if metas else None
        crop_meta = self._con_atributos(crop_meta, crops[0], event.track_id) if crops else crop_meta

        log.info(
            f"📸 Vehículo {event.event_type.value} (track #{event.track_id}, {event.reason}). "
            f"Patente local: {fused.plate or '-'} (apoyo {fused.support}/{fused.n_reads}, "
            f"min {fused.min_char_conf:.2f}{', FUERTE' if fused.is_strong() else ''}); "
            f"{len(reads)}/{len(raw)} recortes con patente, {(time.perf_counter() - t0) * 1000:.0f} ms.",
            extra={"track_id": event.track_id, "local_ocr": fused.to_payload()},
        )
        self.uploader.save_and_dispatch(
            crops=crops,
            track_id=event.track_id,
            event_type=event.event_type.value,
            timestamp=int(wall_ts),
            reason=event.reason,
            plate_crops=plate_crops,
            local_ocr=fused.to_payload() if reads else None,
            vehicle_class=self._vehicle_class(event.track_id),
            crop_meta=crop_meta,
        )

    def _render(self, frame: np.ndarray, boxes, track_ids) -> None:
        """Visualización liviana: se dibuja sobre una copia reducida (no sobre 2880x1620)."""
        h, w = frame.shape[:2]
        scale = cfg.display_width / float(w)
        small = cv2.resize(frame, (cfg.display_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        for (x1, y1, x2, y2), tid in zip(boxes, track_ids):
            p1 = (int(x1 * scale), int(y1 * scale))
            p2 = (int(x2 * scale), int(y2 * scale))
            cv2.rectangle(small, p1, p2, (255, 255, 0), 2)
            cv2.putText(small, f"#{tid}", (p1[0], max(12, p1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        self.last_annotated_frame = self.roi_engine.draw_overlay(small)

    def _inference_loop(self) -> None:
        log.info("Iniciando bucle de inferencia y tracking...")
        last_cleanup_time = time.monotonic()

        while not self.stop_event.is_set():
            # Cerrar tracks perdidos aunque no lleguen frames (evento diferido)
            for ev in self.roi_engine.collect_finished(time.monotonic()):
                self._dispatch(ev, None, time.time())

            try:
                frame, wall_ts, mono_ts = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            self.frame_counter += 1

            # Limpieza periódica de memoria de tracks
            if time.monotonic() - last_cleanup_time > 30.0:
                self.tracker_state.cleanup()
                self.frame_buffer.clear_stale()
                if len(self._last_main_crop) > 200:
                    self._last_main_crop.clear()
                last_cleanup_time = time.monotonic()

            # Saltado de cuadros para optimizar carga de CPU
            if self.frame_counter % cfg.process_every_n_frames != 0:
                continue

            # Pre-filtro de movimiento (MOG2): en reposo y sin vehículos cerca del portón no se corre
            # YOLO (antes un auto estacionado enfrente lo mantenía corriendo todo el día).
            has_motion = self._check_motion(frame)
            if not has_motion and self.roi_engine.relevant_track_count() == 0:
                if cfg.show_display:
                    self._render(frame, [], [])
                continue

            # Tope de detecciones por segundo (los NVR detectan a ~5 fps; un auto en el portón
            # se mueve despacio y el tracker no necesita más)
            now_mono = time.monotonic()
            if cfg.detect_max_fps > 0 and now_mono - self._last_infer_mono < 1.0 / cfg.detect_max_fps:
                continue
            self._last_infer_mono = now_mono

            h, w = frame.shape[:2]

            try:
                t0 = time.perf_counter()
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker=cfg.tracker_name,
                    classes=list(cfg.vehicle_classes),
                    conf=cfg.yolo_confidence,
                    imgsz=cfg.yolo_imgsz,
                    verbose=False,
                )
                self.last_inference_ms = (time.perf_counter() - t0) * 1000
                self.inference_counter += 1
            except Exception as e:
                log.error(f"Error en inferencia YOLO: {e}")
                continue

            boxes: list = []
            track_ids: list = []
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.int().cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                # Qué clase de vehículo es: se manda con el evento para que la tarifa sea la
                # correcta aunque el lector en la nube no responda
                for tid, cls in zip(track_ids, results[0].boxes.cls.int().cpu().tolist()):
                    self._track_cls.setdefault(tid, Counter())[cls] += 1
                if len(self._track_cls) > 500:
                    for viejo in list(self._track_cls)[:250]:
                        del self._track_cls[viejo]

            lighting = analyze_lighting(frame) if boxes else None
            main_frame: Optional[np.ndarray] = None   # se convierte a lo sumo una vez por iteración

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                # detecciones espurias diminutas (2000 px² en 2880x1620, proporcional a la resolución)
                if (x2 - x1) * (y2 - y1) < 0.00043 * w * h:
                    continue

                roi_event = self.roi_engine.update(
                    track_id=track_id, bbox=(x1, y1, x2, y2),
                    frame_height=h, frame_width=w, timestamp=mono_ts,
                )
                if self.main_source is not None:
                    # posición en el sub-stream: se usa para recortar del principal en ese instante
                    self._bbox_hist.setdefault(track_id, deque(maxlen=40)).append((mono_ts, (x1, y1, x2, y2), (h, w)))

                vehicle_crop = None
                # Solo se guardan recortes de vehículos que ya tocaron el portal
                # (evita gastar CPU/memoria en autos estacionados en la calle).
                if self.roi_engine.is_candidate(track_id) or roi_event is not None:
                    src, bx = frame, (x1, y1, x2, y2)
                    if self.main_source is not None:
                        # los recortes salen del principal (_crop_from_main); el del sub-stream
                        # queda solo como respaldo si el evento se despacha sin ninguno
                        m = cfg.crop_margin_px // 4
                        self._sub_fallback[track_id] = frame[max(0, y1 - m):y2 + m, max(0, x1 - m):x2 + m].copy()
                        src = None
                    if src is not None:
                        sh, sw = src.shape[:2]
                        bx1, by1, bx2, by2 = bx
                        # margen extra: la caja viene del sub-stream (menos precisa al escalar)
                        margin = max(cfg.crop_margin_px, int(0.06 * (bx2 - bx1)))
                        vehicle_crop = src[max(0, by1 - margin):min(sh, by2 + margin),
                                           max(0, bx1 - margin):min(sw, bx2 + margin)]
                        if vehicle_crop.size > 0:
                            self.frame_buffer.add(
                                track_id=track_id, frame=None, crop=vehicle_crop,
                                bbox=bx, lighting=lighting, frame_shape=(sh, sw),
                            )

                if roi_event is not None:
                    self._dispatch(roi_event, vehicle_crop, wall_ts)

            if self.main_source is not None:
                self._crop_from_main(lighting)

            if cfg.show_display:
                self._render(frame, boxes, track_ids)

        log.info("Bucle de inferencia terminado.")

    def _crop_from_main(self, lighting) -> None:
        """
        Recorta los vehículos en el portón de cada cuadro nuevo del principal (keyframes), usando la
        posición del sub-stream más cercana en el tiempo (escalada a la resolución del principal).
        """
        for arrival, vframe in self.main_source.pop_new():
            t_main = arrival - cfg.main_stream_offset_sec
            img = None
            for tid, hist in list(self._bbox_hist.items()):
                if not self.roi_engine.is_candidate(tid) or not hist:
                    continue
                # con todos los cuadros (mini PC) no hace falta recortar 15 veces por segundo
                if arrival - self._last_main_crop.get(tid, -1e9) < cfg.main_crop_interval_sec:
                    continue
                ts, (x1, y1, x2, y2), (h, w) = min(hist, key=lambda e: abs(e[0] - t_main))
                if abs(ts - t_main) > cfg.main_match_max_sec:
                    continue
                if img is None:
                    img = vframe.to_ndarray(format="bgr24")
                mh, mw = img.shape[:2]
                sx, sy = mw / float(w), mh / float(h)
                bx1, by1, bx2, by2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
                # margen extra: la caja viene del sub-stream y el vehículo pudo moverse unos cm
                margin = max(cfg.crop_margin_px, int(0.08 * (bx2 - bx1)))
                crop = img[max(0, by1 - margin):min(mh, by2 + margin), max(0, bx1 - margin):min(mw, bx2 + margin)]
                if crop.size > 0:
                    self.frame_buffer.add(track_id=tid, frame=None, crop=crop, bbox=(bx1, by1, bx2, by2),
                                          lighting=lighting, frame_shape=(mh, mw))
                    self._last_main_crop[tid] = arrival
        # memoria acotada: tracks que ya no se ven
        if len(self._bbox_hist) > 50:
            now = time.monotonic()
            for tid in [t for t, hh in self._bbox_hist.items() if hh and now - hh[-1][0] > 30]:
                self._bbox_hist.pop(tid, None)
                self._sub_fallback.pop(tid, None)
