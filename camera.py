import os
import urllib.parse

# Debe setearse ANTES de cargar el backend FFmpeg de OpenCV
# nobuffer/low_delay: evita que FFmpeg acumule video atrasado cuando el CPU no alcanza
# (en la auditoría el stream llegó a ir 17 s detrás de la realidad).
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay",
)

import cv2

from dahua_capture import DahuaSnapshotCapture
from dahua_auth import dahua_credentials_from_env


def build_dahua_sources():
    source = os.getenv("CAMERA_SOURCE", "").strip()
    if source:
        yield source

    host = os.getenv("DAHUA_HOST", "").strip()
    if not host:
        return

    port = os.getenv("DAHUA_PORT", "8081").strip()
    rtsp_port = os.getenv("DAHUA_RTSP_PORT", "554").strip()
    channel = os.getenv("DAHUA_CHANNEL", "1").strip()
    subtype_primary = os.getenv("DAHUA_SUBTYPE_MAIN", "0").strip()
    subtype_secondary = os.getenv("DAHUA_SUBTYPE_SUB", "1").strip()
    user, password = dahua_credentials_from_env()

    if user or password:
        user_escaped = urllib.parse.quote(user, safe="")
        pass_escaped = urllib.parse.quote(password, safe="")
        auth = f"{user_escaped}:{pass_escaped}@"
    else:
        auth = ""

    if rtsp_port:
        yield f"rtsp://{auth}{host}:{rtsp_port}/cam/realmonitor?channel={channel}&subtype={subtype_primary}"
        yield f"rtsp://{auth}{host}:{rtsp_port}/cam/realmonitor?channel={channel}&subtype={subtype_secondary}"
        yield f"rtsp://{auth}{host}:{rtsp_port}/h264/ch{channel}/main/av_stream"
        yield f"rtsp://{auth}{host}:{rtsp_port}/h264/ch{channel}/sub/av_stream"
        yield f"rtsp://{auth}{host}:{rtsp_port}/mpeg4/ch{channel}/main/av_stream"
        yield f"rtsp://{auth}{host}:{rtsp_port}/mpeg4/ch{channel}/sub/av_stream"

    if port:
        yield f"http://{auth}{host}:{port}/cgi-bin/mjpg/video.cgi?channel={channel}&subtype={subtype_secondary}"
        yield f"http://{auth}{host}:{port}/cgi-bin/mjpg/video.cgi?channel={channel}&subtype={subtype_primary}"
        yield f"http://{auth}{host}:{port}/cgi-bin/mjpeg?channel={channel}"
        yield f"http://{auth}{host}:{port}/cgi-bin/mjpeg?channel={channel}&subtype={subtype_secondary}"
        yield f"http://{auth}{host}:{port}/cgi-bin/mjpeg?channel={channel}&subtype={subtype_primary}"
        yield f"http://{auth}{host}:{port}/cam/realmonitor?channel={channel}&subtype={subtype_secondary}"
        yield f"http://{auth}{host}:{port}/cam/realmonitor?channel={channel}&subtype={subtype_primary}"

    if os.getenv("DAHUA_USE_SNAPSHOT", "").strip().lower() in ("1", "true", "yes"):
        yield f"http://{auth}{host}:{port}/cgi-bin/snapshot.cgi?channel={channel}"


def _safe_source_label(source: str) -> str:
    """Oculta credenciales al loguear URLs."""
    if "@" in source:
        return source.split("@", 1)[-1]
    return source


def open_ip_source(source: str):
    # Timeouts explícitos: sin ellos OpenCV espera 30 s por cada URL que no responde.
    open_ms = int(os.getenv("CAMERA_OPEN_TIMEOUT_MS", "8000"))
    read_ms = int(os.getenv("CAMERA_READ_TIMEOUT_MS", "8000"))
    cap = cv2.VideoCapture(
        source,
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_ms],
    )
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None
    return cap


def open_camera(force_snapshot: bool = False):
    """
    Abre webcam USB (CAMERA_INDEX) o stream IP Dahua.
    El uso de snapshot solo se activa si DAHUA_USE_SNAPSHOT está habilitado
    o si force_snapshot es True.
    """
    use_snapshot = force_snapshot or os.getenv("DAHUA_USE_SNAPSHOT", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    if use_snapshot:
        cap = DahuaSnapshotCapture()
        if cap.open():
            label = os.getenv("CAMERA_ID", "dahua_snapshot")
            print("Abierta cámara Dahua en modo SNAPSHOT.")
            return cap, label, True
        if force_snapshot:
            print("Snapshot Dahua forzado falló.")
        else:
            print("DAHUA_USE_SNAPSHOT habilitado pero snapshot no pudo abrirse. Intentando streams IP...")

    for source in build_dahua_sources():
        if not source:
            continue
        # En Picoo el HTTP no sirve; priorizar RTSP y omitir HTTP si ya hay DAHUA_HOST
        if source.startswith("http://") and os.getenv("DAHUA_SKIP_HTTP", "1").strip() in (
            "1",
            "true",
            "yes",
        ):
            continue
        label_src = _safe_source_label(source)
        print(f"Intentando abrir fuente Dahua: {label_src}")
        cap = open_ip_source(source)
        if cap:
            label = os.getenv("CAMERA_ID", label_src)
            print(f"Fuente Dahua abierta: {label}")
            return cap, label, True
        print(f"Fuente Dahua no abrió: {label_src}")

    if os.getenv("DAHUA_HOST", "").strip():
        print("No se pudo abrir ninguna fuente Dahua configurada.")
        if use_snapshot:
            print("DAHUA_USE_SNAPSHOT estaba activado, pero no se pudo iniciar.")
        return cv2.VideoCapture(-1), os.getenv("CAMERA_ID", "dahua_unknown"), True

    index = int(os.getenv("CAMERA_INDEX", 0))
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    label = f"index:{index}"
    return cap, label, False
