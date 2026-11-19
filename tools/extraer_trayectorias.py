"""
Pasa una grabación del sub-stream por el mismo detector y seguimiento que usa el agente (YOLO + ByteTrack,
mismas clases, confianza y resolución) y guarda la posición de cada vehículo en cada cuadro.
Sirve para evaluar reglas del portón (tools/evaluar_reglas.py) sin red ni nube y con los mismos autos.

Uso (desde anpr-edge/):
    venv/Scripts/python.exe tools/extraer_trayectorias.py <sub.mkv> <salida.jsonl> [cada_n_cuadros]

Solo lee el video. Salida: una línea por cuadro procesado {t, h, w, det: [[id, clase, x1, y1, x2, y2, conf], ...]}.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import av

AGENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import cfg  # noqa: E402


def main() -> int:
    src, out = sys.argv[1], sys.argv[2]
    cada = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    from ultralytics import YOLO

    print("cargando modelo", flush=True)
    model = YOLO(str(AGENT / cfg.model_name))
    print("modelo listo", flush=True)
    c = av.open(src)
    s = c.streams.video[0]
    n = 0
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f:
        for fr in c.decode(s):
            n += 1
            if (n - 1) % cada:
                continue
            img = fr.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            r = model.track(img, persist=True, tracker=cfg.tracker_name, classes=list(cfg.vehicle_classes),
                            conf=cfg.yolo_confidence, imgsz=cfg.yolo_imgsz, verbose=False)[0]
            det = []
            if r.boxes is not None and r.boxes.id is not None:
                for (x1, y1, x2, y2), tid, cl, cf in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(),
                                                         r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                    if (x2 - x1) * (y2 - y1) < 0.00043 * w * h:   # mismo filtro de detecciones diminutas que el agente
                        continue
                    det.append([tid, cl, round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1), round(cf, 3)])
            f.write(json.dumps({"t": round(float(fr.pts * s.time_base), 3), "h": h, "w": w, "det": det}) + "\n")
            f.flush()
            if n % 300 == 0:
                print(f"{n} cuadros, {time.time() - t0:.0f} s", flush=True)
    print(f"FIN {n} cuadros en {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
