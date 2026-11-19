"""
Punto de entrada principal para el Agente Edge de Reconocimiento de Vehículos y Placas (ANPR 2026).
Arquitectura Híbrida Edge-Cloud para estacionamientos con puerta única bidireccional sin barrera.
"""
from __future__ import annotations

import os
import queue
import signal
import time

import cv2

from cloud.offline_queue import OfflineQueue
from cloud.uploader import CloudUploader
from config import cfg
from core.frame_selector import FrameBuffer
from core.image_quality import ANPRImageEnhancer
from core.plate_reader import PlateReader
from core.roi_engine import ROIEngine
from core.tracker_state import TrackerState
from core.quietos import RegistroQuietos
from core.repetidos import RegistroRepetidos
from pipeline.consumer import InferenceConsumer
from pipeline.dual_producer import DualStreamProducer
from pipeline.producer import FrameProducer
from utils.logger import get_logger

log = get_logger("main")


def main() -> None:
    log.info("==========================================================")
    log.info("🚀 Iniciando Agente ANPR Edge (Estacionamiento 2026)")
    log.info(f"📍 Sucursal ID: {cfg.default_sucursal_id}")
    log.info(f"📷 Cámara ID: {cfg.camera_id}")
    log.info(f"🧠 Modelo YOLO: {cfg.model_name} (Imgsz: {cfg.yolo_imgsz})")
    log.info(f"⚡ ROI Config: Line A (portal/rampa) = {cfg.roi_line_a_y_ratio}, Line B (calle/portal) = {cfg.roi_line_b_y_ratio}")
    log.info("==========================================================")

    # 1. Inicializar componentes centrales
    offline_queue = OfflineQueue(db_path=cfg.offline_queue_db_path)
    uploader = CloudUploader(offline_queue=offline_queue)
    tracker_state = TrackerState(
        track_id_ttl_sec=cfg.max_processed_ids_ttl,
        plate_cooldown_sec=cfg.cooldown_per_plate_sec,
        max_active=cfg.max_active_tracks,
    )
    roi_engine = ROIEngine(
        line_a_y_ratio=cfg.roi_line_a_y_ratio,
        line_b_y_ratio=cfg.roi_line_b_y_ratio,
        direction_timeout_sec=cfg.roi_direction_timeout_sec,
        polygon_ratios=cfg.roi_polygon_ratios,
        track_lost_sec=cfg.roi_track_lost_sec,
        min_displacement_ratio=cfg.roi_min_displacement_ratio,
        continuation_sec=cfg.roi_continuation_sec,
        event_hold_sec=cfg.roi_event_hold_sec,
        solo_movimiento=cfg.roi_solo_movimiento in ("true", "1", "yes", "sombra"),
        solo_cruces=cfg.roi_solo_cruces in ("true", "1", "yes", "sombra"),
        union_sec=cfg.roi_union_sec,
    )
    frame_buffer = FrameBuffer(
        max_size=cfg.frame_buffer_size,
        min_blur=cfg.min_blur_threshold,
    )
    enhancer = ANPRImageEnhancer()
    plate_reader = (
        PlateReader.try_create(cfg.plate_detector_model, cfg.plate_ocr_model, threads=cfg.plate_reader_threads)
        if cfg.plate_reader_enabled else None
    )

    # 2. Cola compartida de frames (Drop-Oldest, size 1)
    frame_queue: queue.Queue = queue.Queue(maxsize=1)

    # 3. Hilos de captura e inferencia
    detect_url = cfg.detect_source or (
        os.getenv("CAMERA_SOURCE", "").replace("subtype=0", "subtype=1")
        if "subtype=0" in os.getenv("CAMERA_SOURCE", "") else ""
    )
    if cfg.dual_stream_enabled and detect_url:
        producer = DualStreamProducer(frame_queue, detect_url=detect_url, main_url=os.getenv("CAMERA_SOURCE", ""))
        main_source = producer.main
        log.info("Modo dos streams: detección en el sub-stream, recortes del principal.")
    else:
        producer = FrameProducer(frame_queue=frame_queue)
        main_source = None
    uploader.status_provider = producer.status
    # Regla estricta (ROI_SOLO_CRUCES) manda sobre ROI_SOLO_MOVIMIENTO: todo lo que no cruzó se archiva
    estricta = cfg.roi_solo_cruces in ("true", "1", "yes", "sombra")
    modo_quietos = cfg.roi_solo_cruces if estricta else cfg.roi_solo_movimiento
    quietos = (RegistroQuietos(cfg.quietos_db_path, cfg.quietos_dir, modo_quietos,
                                min_repeticiones=cfg.quietos_min_repeticiones, estricto=estricta)
               if estricta or cfg.roi_solo_movimiento in ("true", "1", "yes", "sombra") else None)
    repetidos = (RegistroRepetidos(cfg.quietos_db_path, cfg.repetidos_dir, cfg.roi_agrupar_repetidos)
                 if cfg.roi_agrupar_repetidos in ("true", "1", "yes", "sombra") else None)
    consumer = InferenceConsumer(
        frame_queue=frame_queue,
        uploader=uploader,
        tracker_state=tracker_state,
        roi_engine=roi_engine,
        frame_buffer=frame_buffer,
        enhancer=enhancer,
        plate_reader=plate_reader,
        main_source=main_source,
        quietos=quietos,
        repetidos=repetidos,
    )

    # 4. Manejo de terminación por señales
    is_running = True

    def signal_handler(sig, frame):
        nonlocal is_running
        log.info("Señal de interrupción recibida. Apagando componentes de forma segura...")
        is_running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 5. Iniciar pipeline
    if not producer.start():
        log.critical("Fallo al iniciar el productor de video. Abortando.")
        return

    if not consumer.start():
        log.critical("Fallo al iniciar el consumidor de inferencia. Abortando.")
        producer.stop()
        return

    log.info("✅ Pipeline ANPR Edge en ejecución normal.")

    window_name = "Sistema ANPR 2026 - Control de Acceso"
    if cfg.show_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, cfg.display_width, int(cfg.display_width * 9 / 16))

    # 6. Bucle de supervisión y visualización
    try:
        while is_running:
            if cfg.show_display:
                display_frame = consumer.last_annotated_frame
                if display_frame is not None:
                    cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(20) & 0xFF
                if key == ord("q"):
                    log.info("Salida manual solicitada mediante tecla 'q'.")
                    break
            else:
                time.sleep(0.5)

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt recibido.")
    finally:
        log.info("Deteniendo hilos de procesamiento...")
        producer.stop()
        consumer.stop()
        uploader.stop()

        if cfg.show_display:
            cv2.destroyAllWindows()

        log.info("🔒 Sistema ANPR Edge apagado correctamente. Recursos liberados.")


if __name__ == "__main__":
    main()
