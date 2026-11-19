"""
Configuración centralizada del Agente ANPR Edge (2026).
Todas las variables de entorno se validan y tipan aquí.
Importar `cfg` desde cualquier módulo del proyecto.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Cargar SIEMPRE el .env junto a este archivo, sin importar desde dónde se ejecute
# el agente (antes se buscaba desde el script llamador y podía abrirse la webcam local).
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class ANPRConfig:
    # ── Supabase / Cloud ────────────────────────────────────────────────────
    supabase_url: str
    supabase_key: str
    webhook_url: str
    default_sucursal_id: str
    camera_id: str

    # ── YOLO / Modelo ───────────────────────────────────────────────────────
    model_name: str
    tracker_name: str
    yolo_confidence: float
    yolo_imgsz: int
    vehicle_classes: tuple[int, ...]
    process_every_n_frames: int

    # ── ROI Bidireccional (zonas por coordenada Y del borde inferior del bbox) ─
    # Geometría: cámara DENTRO del garaje apuntando hacia la calle.
    #   y < Línea B            → CALLE (tráfico que pasa, autos estacionados enfrente)
    #   Línea B ≤ y < Línea A  → PORTAL (portón / vereda: mejor momento para la patente)
    #   y ≥ Línea A            → INTERIOR (rampa, cerca de la cámara)
    #   ENTRADA: el vehículo avanza hacia la cámara (y crece).
    #   SALIDA:  el vehículo se aleja de la cámara (y decrece).
    roi_line_a_y_ratio: float       # Línea A: límite portal/interior (≈ 0.68)
    roi_line_b_y_ratio: float       # Línea B: límite calle/portal  (≈ 0.56)
    roi_direction_timeout_sec: float
    roi_track_lost_sec: float       # Sin ver el track este tiempo → se evalúa y cierra
    roi_min_displacement_ratio: float  # Desplazamiento vertical mínimo para inferir dirección
    roi_continuation_sec: float     # Ventana para tratar un ID nuevo como el mismo vehículo
    roi_event_hold_sec: float       # Espera tras decidir la dirección para juntar fotos más cercanas
    roi_solo_movimiento: str        # "false" | "sombra" | "true" (ver core/quietos.py)
    roi_solo_cruces: str            # "false" | "sombra" | "true": regla estricta, solo el cruce de línea genera aviso
    roi_union_sec: float            # regla estricta: espera para unir un seguimiento cortado a mitad del portón
    quietos_db_path: str            # registro permanente de objetos quietos (nunca se borra)
    roi_agrupar_repetidos: str      # "false" | "sombra" | "true" (ver core/repetidos.py)
    repetidos_dir: str              # fotos de los avisos repetidos agrupados o en sombra (nunca se borran)
    quietos_min_repeticiones: int   # avisos iguales previos para suprimir un quieto (0 = sin cruce de línea no hay aviso)
    quietos_dir: str                # fotos de los quietos suprimidos o en sombra (nunca se borran)

    # Polígono ROI opcional para ignorar laterales del frame.
    # Formato: "x1,y1,x2,y2,x3,y3,x4,y4" (porcentajes 0..1)
    # Si vacío, se usa el frame completo.
    roi_polygon_ratios: str

    # ── Calidad de imagen ───────────────────────────────────────────────────
    min_blur_threshold: float       # Laplacian variance mínima aceptable
    min_crop_width: int             # Ancho mínimo del crop en píxeles
    min_crop_height: int            # Alto mínimo del crop en píxeles
    min_bbox_area: int              # Área mínima del bounding box (px²)
    frame_buffer_size: int          # Frames en buffer para best-frame selection
    crop_margin_px: int             # Margen extra alrededor del bounding box
    crops_per_event: int            # Recortes enviados por evento (lectura multi-frame)

    # ── Lectura local de patentes (core/plate_reader.py) ────────────────────
    plate_reader_enabled: bool      # Detector + OCR de patente en el portón (segunda opinión de Gemini)
    plate_detector_model: str
    plate_ocr_model: str            # cct-s-v2-global: 6/7 patentes legibles en el set real del portón
    plate_crops_per_event: int      # Patentes ampliadas que se envían a la nube
    plate_reader_threads: int       # Hilos ONNX del lector (0 = automático según núcleos)

    # ── Anti-flood / Cooldown ────────────────────────────────────────────────
    cooldown_per_plate_sec: int     # Cooldown por patente+dirección
    max_processed_ids_ttl: int      # TTL de track IDs procesados (segundos)
    max_active_tracks: int          # Límite de tracks activos simultáneos

    # ── Cámara / Reconexión ─────────────────────────────────────────────────
    reconnect_delay_sec: float
    ip_read_failures_before_reconnect: int
    max_reconnects_before_exit: int
    proactive_reconnect_interval_sec: float  # 0 = desactivado
    camera_open_timeout_ms: int
    camera_read_timeout_ms: int

    # ── Dos streams (detección en el sub-stream, recortes del principal) ────
    dual_stream_enabled: bool
    detect_source: str              # URL del sub-stream (vacío = CAMERA_SOURCE con subtype=1)
    detect_max_fps: float           # Tope de inferencias YOLO por segundo (0 = sin tope: lo que dé el equipo)
    main_decode_mode: str           # "auto" | "all" | "keyframes" (ver pipeline/dual_producer.py)
    main_crop_interval_sec: float   # Con "all": cada cuánto se recorta del principal un vehículo
    main_stream_offset_sec: float   # Desfase medido entre la llegada del sub-stream y el principal
    main_match_max_sec: float       # Distancia máxima entre el cuadro del principal y la posición del sub-stream

    # ── Performance ─────────────────────────────────────────────────────────
    use_threading: bool             # Producer-consumer threads separados
    use_mog2_prefilter: bool        # Background subtraction pre-YOLO
    mog2_motion_threshold: float    # % mínimo de píxeles en movimiento (0..1)

    # ── UI / Debug ──────────────────────────────────────────────────────────
    show_display: bool              # Mostrar ventana cv2.imshow
    display_width: int              # Ancho de la ventana de visualización
    log_level: str                  # DEBUG | INFO | WARNING | ERROR

    # ── Cola Offline (SQLite) ────────────────────────────────────────────────
    offline_queue_db_path: str
    offline_retry_interval_sec: float
    offline_max_retries: int
    offline_max_age_days: float     # Un evento sin subir se reintenta hasta esta antigüedad (nunca se descarta antes)

    # ── Mantenimiento ────────────────────────────────────────────────────────
    heartbeat_interval_sec: float   # Consulta liviana a Supabase para evitar la pausa por inactividad (0 = off)
    agent_status_interval_sec: float  # Latido al panel (estado de cámara y cola) vía la Edge Function (0 = off)
    agent_secret: str               # Secreto compartido con la Edge Function (header x-agent-secret)
    webhook_timeout_sec: float      # Espera de la Edge Function (OCR incluido) antes de reintentar

    @classmethod
    def from_env(cls) -> "ANPRConfig":
        return cls(
            # Supabase
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_key=os.getenv("SUPABASE_KEY", ""),
            webhook_url=os.getenv("WEBHOOK_URL", ""),
            default_sucursal_id=os.getenv("DEFAULT_SUCURSAL_ID", "unknown-sucursal"),
            camera_id=os.getenv("CAMERA_ID", "edge_cam"),

            # YOLO (nano por defecto: en la auditoría yolo11s tardaba ~1.6 s por frame en CPU)
            model_name=os.getenv("YOLO_MODEL", "yolov8n.pt"),
            # ByteTrack: BoT-SORT calculaba el movimiento de cámara (GMC) en cada cuadro, +120 ms en
            # el Pentium, y esta cámara es fija
            tracker_name=os.getenv("YOLO_TRACKER", "bytetrack.yaml"),
            yolo_confidence=float(os.getenv("YOLO_CONFIDENCE", "0.40")),
            yolo_imgsz=int(os.getenv("YOLO_IMGSZ", "320")),
            vehicle_classes=(2, 3, 5, 7),  # car, motorcycle, bus, truck
            process_every_n_frames=max(1, int(os.getenv("PROCESS_EVERY_N_FRAMES", "1"))),

            # ROI
            roi_line_a_y_ratio=float(os.getenv("ROI_LINE_A_Y_RATIO", "0.68")),
            roi_line_b_y_ratio=float(os.getenv("ROI_LINE_B_Y_RATIO", "0.56")),
            roi_direction_timeout_sec=float(os.getenv("ROI_DIRECTION_TIMEOUT_SEC", "8.0")),
            roi_track_lost_sec=float(os.getenv("ROI_TRACK_LOST_SEC", "3.5")),
            roi_min_displacement_ratio=float(os.getenv("ROI_MIN_DISPLACEMENT_RATIO", "0.04")),
            roi_continuation_sec=float(os.getenv("ROI_CONTINUATION_SEC", "8.0")),
            roi_event_hold_sec=float(os.getenv("ROI_EVENT_HOLD_SEC", "1.5")),
            roi_solo_movimiento=os.getenv("ROI_SOLO_MOVIMIENTO", "false").strip().lower(),
            roi_solo_cruces=os.getenv("ROI_SOLO_CRUCES", "false").strip().lower(),
            roi_union_sec=float(os.getenv("ROI_UNION_SEC", "3.0")),
            quietos_db_path=os.getenv("QUIETOS_DB", str(BASE_DIR / "quietos.db")),
            quietos_dir=os.getenv("QUIETOS_DIR", str(BASE_DIR / "quietos")),
            quietos_min_repeticiones=int(os.getenv("QUIETOS_MIN_REPETICIONES", "5")),
            roi_agrupar_repetidos=os.getenv("ROI_AGRUPAR_REPETIDOS", "false").strip().lower(),
            repetidos_dir=os.getenv("REPETIDOS_DIR", str(BASE_DIR / "repetidos")),
            roi_polygon_ratios=os.getenv("ROI_POLYGON_RATIOS", ""),

            # Image quality
            min_blur_threshold=float(os.getenv("MIN_BLUR_THRESHOLD", "60.0")),
            min_crop_width=int(os.getenv("MIN_CROP_WIDTH", "200")),
            min_crop_height=int(os.getenv("MIN_CROP_HEIGHT", "150")),
            min_bbox_area=int(os.getenv("MIN_BBOX_AREA", "15000")),
            frame_buffer_size=int(os.getenv("FRAME_BUFFER_SIZE", "8")),
            crop_margin_px=int(os.getenv("CROP_MARGIN_PX", "20")),
            crops_per_event=max(1, int(os.getenv("CROPS_PER_EVENT", "3"))),

            # Lectura local de patentes (modelos ONNX; se descargan la primera vez)
            plate_reader_enabled=_env_bool("PLATE_READER_ENABLED", "true"),
            plate_detector_model=os.getenv("PLATE_DETECTOR_MODEL", "yolo-v9-t-384-license-plate-end2end"),
            plate_ocr_model=os.getenv("PLATE_OCR_MODEL", "cct-s-v2-global-model"),
            plate_crops_per_event=max(1, int(os.getenv("PLATE_CROPS_PER_EVENT", "2"))),
            plate_reader_threads=max(0, int(os.getenv("PLATE_READER_THREADS", "0"))),

            # Anti-flood
            cooldown_per_plate_sec=int(os.getenv("COOLDOWN_PER_PLATE_SEC", "45")),
            max_processed_ids_ttl=int(os.getenv("MAX_PROCESSED_IDS_TTL", "300")),
            max_active_tracks=int(os.getenv("MAX_ACTIVE_TRACKS", "20")),

            # Camera / Connection
            reconnect_delay_sec=float(os.getenv("RECONNECT_DELAY_SEC", "3.0")),
            ip_read_failures_before_reconnect=int(os.getenv("IP_READ_FAILURES", "3")),
            max_reconnects_before_exit=int(os.getenv("MAX_RECONNECTS", "10")),
            proactive_reconnect_interval_sec=float(os.getenv("PROACTIVE_RECONNECT_SEC", "0")),
            camera_open_timeout_ms=int(os.getenv("CAMERA_OPEN_TIMEOUT_MS", "8000")),
            camera_read_timeout_ms=int(os.getenv("CAMERA_READ_TIMEOUT_MS", "8000")),

            # Dos streams (práctica estándar de NVR: detectar en baja resolución)
            dual_stream_enabled=_env_bool("DUAL_STREAM", "true"),
            detect_source=os.getenv("CAMERA_SOURCE_DETECT", "").strip(),
            detect_max_fps=float(os.getenv("DETECT_MAX_FPS", "0")),
            main_decode_mode=os.getenv("MAIN_DECODE_MODE", "auto").strip().lower(),
            main_crop_interval_sec=float(os.getenv("MAIN_CROP_INTERVAL_SEC", "0.25")),
            main_stream_offset_sec=float(os.getenv("MAIN_STREAM_OFFSET_SEC", "0.0")),
            main_match_max_sec=float(os.getenv("MAIN_MATCH_MAX_SEC", "0.35")),

            # Performance
            use_threading=_env_bool("USE_THREADING", "true"),
            use_mog2_prefilter=_env_bool("USE_MOG2_PREFILTER", "true"),
            mog2_motion_threshold=float(os.getenv("MOG2_MOTION_THRESHOLD", "0.005")),

            # UI
            show_display=_env_bool("SHOW_DISPLAY", "false"),
            display_width=int(os.getenv("DISPLAY_WIDTH", "960")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),

            # Offline queue
            offline_queue_db_path=os.getenv("OFFLINE_QUEUE_DB", str(BASE_DIR / "offline_queue.db")),
            offline_retry_interval_sec=float(os.getenv("OFFLINE_RETRY_SEC", "15.0")),
            offline_max_retries=int(os.getenv("OFFLINE_MAX_RETRIES", "500")),
            offline_max_age_days=float(os.getenv("OFFLINE_MAX_AGE_DAYS", "14")),

            # Mantenimiento
            heartbeat_interval_sec=float(os.getenv("HEARTBEAT_SEC", "21600")),
            agent_status_interval_sec=float(os.getenv("AGENT_STATUS_SEC", "60")),
            agent_secret=os.getenv("ANPR_AGENT_SECRET", ""),
            # 60 s cortaba lecturas en curso y el reintento procesaba el mismo evento en paralelo
            webhook_timeout_sec=float(os.getenv("WEBHOOK_TIMEOUT_SEC", "150")),
        )


# ── Instancia global ────────────────────────────────────────────────────────
cfg = ANPRConfig.from_env()
