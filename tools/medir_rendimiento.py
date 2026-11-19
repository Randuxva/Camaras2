"""
Mide en ESTA PC cuánto tarda y cuánta memoria usa el detector de vehículos con distintos modelos y
resoluciones, sobre un clip real del portón, y recomienda la combinación de mayor calidad que la
PC aguanta con margen. No cambia nada del agente ni del .env.

Uso (con el agente DETENIDO, desde la carpeta del agente):
    .\\venv\\Scripts\\python.exe tools\\medir_rendimiento.py clip_prueba.mp4

Qué mide por cada combinación (modelo, resolución de detección):
  - ms por cuadro (mediana y percentil 95)           -> cuántos cuadros por segundo alcanza
  - memoria (RAM) que ocupa el proceso               -> la PC tiene poca RAM
  - identificadores de seguimiento creados            -> menos = el seguimiento se corta menos
  - cuadros por vehículo seguido (promedio)           -> más = seguimiento más continuo

Un modelo/resolución mejor no cambia la lectura de la patente (eso depende de la imagen), pero un
detector más preciso pierde menos el seguimiento de los vehículos.
"""
from __future__ import annotations

import gc
import statistics as st
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

AGENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import cfg  # noqa: E402

# Presupuesto: el agente también decodifica video y lee patentes en la misma PC.
PRESUPUESTO_P95_MS = 160        # ~6 cuadros por segundo en el peor momento
PRESUPUESTO_RAM_MB = 900        # el proceso completo del agente ronda los 300 MB sin el detector grande
CANDIDATOS = [
    ("yolov8n.pt", 320),        # el que se usa hoy
    ("yolov8n.pt", 480),
    ("yolov8n.pt", 640),
    ("yolo11s.pt", 320),
    ("yolo11s.pt", 480),
    ("yolo11s.pt", 640),
]


def leer_clip(ruta: str, max_cuadros: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(ruta)
    cuadros = []
    while len(cuadros) < max_cuadros:
        ok, f = cap.read()
        if not ok:
            break
        cuadros.append(f)
    cap.release()
    return cuadros


def medir(modelo: str, imgsz: int, cuadros: list[np.ndarray]) -> dict | None:
    from ultralytics import YOLO

    ruta = AGENT / modelo
    if not ruta.exists():
        return None
    proc = psutil.Process()
    gc.collect()
    ram0 = proc.memory_info().rss / 1e6
    m = YOLO(str(ruta))
    # calentamiento (la primera inferencia es lenta y no representa el uso normal)
    for f in cuadros[:3]:
        m.track(f, persist=True, tracker=cfg.tracker_name, classes=list(cfg.vehicle_classes), imgsz=imgsz,
                conf=cfg.yolo_confidence, verbose=False)
    m.predictor = None
    ms, ids_por_cuadro = [], []
    cpu0 = time.process_time()
    t_inicio = time.perf_counter()
    for f in cuadros:
        t0 = time.perf_counter()
        r = m.track(f, persist=True, tracker=cfg.tracker_name, classes=list(cfg.vehicle_classes), imgsz=imgsz,
                    conf=cfg.yolo_confidence, verbose=False)[0]
        ms.append((time.perf_counter() - t0) * 1000)
        ids = [] if r.boxes is None or r.boxes.id is None else [int(i) for i in r.boxes.id.tolist()]
        ids_por_cuadro.append(ids)
    pared = time.perf_counter() - t_inicio
    cpu = time.process_time() - cpu0
    ram = proc.memory_info().rss / 1e6
    vistos: dict[int, int] = {}
    for ids in ids_por_cuadro:
        for i in ids:
            vistos[i] = vistos.get(i, 0) + 1
    ms_ord = sorted(ms)
    res = {
        "modelo": modelo, "imgsz": imgsz,
        "mediana_ms": st.median(ms), "p95_ms": ms_ord[int(0.95 * (len(ms) - 1))],
        "fps": len(ms) / pared, "cpu_pct": 100.0 * cpu / pared,
        "ram_mb": ram, "ram_extra_mb": ram - ram0,
        "ids": len(vistos), "cuadros_por_id": (sum(vistos.values()) / len(vistos)) if vistos else 0.0,
        "detecciones": sum(vistos.values()),
    }
    del m
    gc.collect()
    return res


LECTORES = [
    ("yolo-v9-t-384-license-plate-end2end", "cct-s-v2-global-model"),     # el que se usa hoy
    ("yolo-v9-t-640-license-plate-end2end", "cct-s-v2-global-model"),
    ("yolo-v9-s-608-license-plate-end2end", "cct-s-v2-global-model"),
    ("yolo-v9-t-384-license-plate-end2end", "argentinian-plates-cnn-model"),
    ("yolo-v9-t-640-license-plate-end2end", "argentinian-plates-cnn-model"),
    ("yolo-v9-t-384-license-plate-end2end", "global-plates-mobile-vit-v2-model"),
]
PRESUPUESTO_LECTURA_MS = 700      # por foto de vehículo; el lector corre aparte y no puede atrasar la detección


def recortes_de_vehiculos(cuadros, maximo=40):
    """Recortes de vehículos del clip (con el detector actual), solo para medir TIEMPO de los lectores."""
    from ultralytics import YOLO

    m = YOLO(str(AGENT / "yolov8n.pt"))
    out = []
    for f in cuadros[::6]:
        r = m.predict(f, classes=list(cfg.vehicle_classes), imgsz=320, conf=cfg.yolo_confidence, verbose=False)[0]
        for b in (r.boxes.xyxy.tolist() if r.boxes is not None else []):
            x1, y1, x2, y2 = [int(v) for v in b]
            c = f[max(0, y1):y2, max(0, x1):x2]
            if c.shape[0] > 40 and c.shape[1] > 40:
                out.append(c)
        if len(out) >= maximo:
            break
    return out[:maximo]


def medir_lectores(cuadros) -> None:
    """Velocidad y memoria de cada combinación (detector de patente + lector de caracteres) en ESTA PC."""
    from core.plate_reader import PlateReader

    crops = recortes_de_vehiculos(cuadros)
    if not crops:
        print("\nNo encontré vehículos en el clip para medir los lectores de patentes.")
        return
    proc = psutil.Process()
    print(f"\n--- Lectores de patentes: {len(crops)} recortes de vehículos ---")
    print(f"{'detector':14s} {'lector de caracteres':34s} {'ms/foto':>8s} {'p95':>6s} {'RAM MB':>7s}")
    for det, ocr in LECTORES:
        try:
            gc.collect()
            lector = PlateReader(det, ocr)
        except Exception as e:  # sin internet la primera vez, o modelo no disponible
            print(f"{det.split('-license')[0]:14s} {ocr:34s}  no se pudo cargar: {str(e)[:60]}")
            continue
        lector.read_many(crops[:2])           # calentamiento
        ms = []
        for c in crops:
            t0 = time.perf_counter()
            lector.read_many([c])
            ms.append((time.perf_counter() - t0) * 1000)
        ms.sort()
        marca = "  <-- hoy" if (det, ocr) == LECTORES[0] else ""
        print(f"{det.split('-license')[0]:14s} {ocr.replace('-model', ''):34s} {st.median(ms):7.0f}  {ms[int(0.95 * (len(ms) - 1))]:5.0f} {proc.memory_info().rss / 1e6:7.0f}{marca}", flush=True)
        del lector
    print(f"\nPresupuesto del lector: hasta {PRESUPUESTO_LECTURA_MS} ms por foto (corre en un hilo aparte).")
    print("La PRECISION de cada combinacion se midio en otra PC con fotos reales: ver el informe.")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 450
    cuadros = leer_clip(sys.argv[1], n)
    if not cuadros:
        print("No pude leer el clip:", sys.argv[1])
        return 1
    h, w = cuadros[0].shape[:2]
    print(f"PC: {psutil.cpu_count(logical=False)} núcleos físicos / {psutil.cpu_count()} hilos, "
          f"RAM total {psutil.virtual_memory().total / 1e9:.1f} GB, libre {psutil.virtual_memory().available / 1e9:.1f} GB")
    print(f"Clip: {len(cuadros)} cuadros de {w}x{h}\n")
    print(f"{'modelo':12s} {'imgsz':>5s} {'mediana':>8s} {'p95':>7s} {'fps':>5s} {'CPU%':>5s} {'RAM MB':>7s} {'ids':>4s} {'cuad/id':>8s}")
    filas = []
    for modelo, imgsz in CANDIDATOS:
        r = medir(modelo, imgsz, cuadros)
        if r is None:
            print(f"{modelo:12s} {imgsz:5d}  (no está el archivo del modelo)")
            continue
        filas.append(r)
        print(f"{r['modelo']:12s} {r['imgsz']:5d} {r['mediana_ms']:7.0f}ms {r['p95_ms']:6.0f}ms {r['fps']:5.1f} "
              f"{r['cpu_pct']:5.0f} {r['ram_mb']:7.0f} {r['ids']:4d} {r['cuadros_por_id']:8.1f}", flush=True)
    ok = [r for r in filas if r["p95_ms"] <= PRESUPUESTO_P95_MS and r["ram_mb"] <= PRESUPUESTO_RAM_MB]
    print(f"\nPresupuesto: p95 <= {PRESUPUESTO_P95_MS} ms y RAM del proceso <= {PRESUPUESTO_RAM_MB} MB")
    medir_lectores(cuadros)
    if not ok:
        print("Ninguna combinación cumple el presupuesto: conviene dejar la configuración actual (yolov8n, 320).")
        return 0
    # La mejor = la que sigue mejor a los vehículos (más cuadros por identificador) y, a igualdad, la más rápida.
    mejor = max(ok, key=lambda r: (round(r["cuadros_por_id"], 1), r["detecciones"], -r["p95_ms"]))
    print(f"\nRECOMENDADA: {mejor['modelo']} con resolución {mejor['imgsz']}")
    print("  en el .env del agente:")
    print(f'    YOLO_MODEL="{mejor["modelo"]}"')
    print(f'    YOLO_IMGSZ="{mejor["imgsz"]}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
