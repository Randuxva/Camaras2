"""
Graba el stream principal y el sub-stream de la cámara a la vez, sin recodificar (copia de
paquetes, casi sin CPU), con la hora de llegada de cada cuadro para poder alinearlos.
Sirve para reproducir exactamente los mismos autos al comparar versiones del pipeline.

Uso (desde anpr-edge/): venv/Scripts/python.exe tools/record_streams.py <segundos> [dir]
Salida: <dir>/main.mkv, <dir>/sub.mkv, <dir>/main.jsonl, <dir>/sub.jsonl (pts -> hora de llegada)
Las grabaciones tienen patentes (datos personales): quedan en audit_runs/, fuera de git.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import av

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BASE_DIR  # noqa: E402,F401  (carga el .env)


def record(url: str, out: Path, secs: float) -> None:
    src = av.open(url, options={"rtsp_transport": "tcp"}, timeout=10)
    s_in = src.streams.video[0]
    dst = av.open(str(out.with_suffix(".mkv")), "w")
    s_out = dst.add_stream_from_template(s_in)
    t0 = time.time()
    n = 0
    with open(out.with_suffix(".jsonl"), "w") as log:
        for pkt in src.demux(s_in):
            if pkt.dts is None:
                continue
            if n == 0 and not pkt.is_keyframe:
                continue  # empezar en un keyframe: antes no hay cuadros decodificables
            now = time.time()
            log.write(json.dumps({"pts": pkt.pts, "wall": round(now, 4), "key": bool(pkt.is_keyframe)}) + "\n")
            pkt.stream = s_out
            dst.mux(pkt)
            n += 1
            if now - t0 > secs:
                break
    dst.close()
    src.close()
    print(f"{out.name}: {n} cuadros en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 600
    d = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE_DIR / "audit_runs" / time.strftime("grab_%Y%m%d_%H%M")
    d.mkdir(parents=True, exist_ok=True)
    main = os.getenv("CAMERA_SOURCE", "")
    sub = main.replace("subtype=0", "subtype=1")
    ths = [threading.Thread(target=record, args=(u, d / name, secs)) for u, name in ((main, "main"), (sub, "sub"))]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    print("ok", d)
