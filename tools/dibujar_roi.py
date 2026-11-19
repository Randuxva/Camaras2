"""
Toma un cuadro de la cámara y dibuja encima las líneas A/B y, si se pasa, un polígono de zona (ROI_POLYGON_RATIOS),
para ver QUÉ área se vigilaría antes de tocar el .env. No cambia nada del agente.

    python tools/dibujar_roi.py salida.jpg
    python tools/dibujar_roi.py salida.jpg --poligono "0.09,0.40,0.62,0.40,0.62,0.66,1,0.72,1,1,0,1,0,0.70,0.09,0.62"

El polígono se escribe como x1,y1,x2,y2,... en proporciones 0..1 del cuadro (0,0 = arriba a la izquierda).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from urllib.parse import quote

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg  # noqa: E402  (carga el .env)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("salida")
    ap.add_argument("--poligono", default="")
    a = ap.parse_args()

    h, p = os.getenv("DAHUA_HOST"), os.getenv("DAHUA_RTSP_PORT")
    u, w = os.getenv("DAHUA_USER", ""), os.getenv("DAHUA_PASS", "")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;8000000"
    cap = cv2.VideoCapture(f"rtsp://{quote(u)}:{quote(w)}@{h}:{p}/cam/realmonitor?channel=1&subtype=0", cv2.CAP_FFMPEG)  # secret-scan: ignore (usuario y clave salen del .env, no están escritos acá)
    frame, t0, n = None, time.time(), 0
    while time.time() - t0 < 40 and n < 3:
        ok, f = cap.read()
        if ok and f is not None:
            frame, n = f, n + 1
    cap.release()
    if frame is None:
        print("No se pudo leer la cámara.")
        return 1

    H, W = frame.shape[:2]
    g = frame.copy()
    if a.poligono:
        v = [float(x) for x in a.poligono.split(",")]
        pts = np.array([(int(v[i] * W), int(v[i + 1] * H)) for i in range(0, len(v), 2)], np.int32)
        capa = g.copy()
        cv2.fillPoly(capa, [pts], (0, 255, 0))
        g = cv2.addWeighted(capa, 0.25, g, 0.75, 0)      # verde = zona que se vigila
        cv2.polylines(g, [pts], True, (0, 255, 0), 8)
    for y, color, nombre in ((cfg.roi_line_a_y_ratio, (0, 255, 0), "A"), (cfg.roi_line_b_y_ratio, (0, 200, 255), "B")):
        cv2.line(g, (0, int(H * y)), (W, int(H * y)), color, 6)
        cv2.putText(g, nombre, (20, int(H * y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
    cv2.imwrite(a.salida, cv2.resize(g, (1280, int(1280 * H / W))))
    print(f"Guardado {a.salida} ({W}x{H})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
