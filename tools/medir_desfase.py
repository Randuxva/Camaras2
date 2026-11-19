"""
Mide el desfase entre el sub-stream (donde el agente detecta) y el stream principal (de donde recorta).

El agente ubica cada vehículo en el sub-stream y lo recorta del principal en "ese mismo instante"
(MAIN_STREAM_OFFSET_SEC). Si el principal llega con otro retraso, el recorte sale de otro momento:
moto a la mitad, ventanilla en vez de patente, patente fuera de cuadro.

Cómo mide: abre los dos streams SIN decodificar (solo paquetes, casi sin CPU) y anota cuándo llega
cada paquete y cuántos bytes pesa. Cuando algo se mueve en la escena los paquetes pesan más en los
dos streams; correlacionando las dos series se obtiene cuánto después llega el movimiento en uno que
en el otro. Hace falta movimiento real en la escena (autos pasando) durante la medición.

    python tools/medir_desfase.py                 # 5 minutos
    python tools/medir_desfase.py --segundos 600

No imprime ni guarda credenciales. Deja un resumen en audit_runs/desfase_<fecha>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

AGENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from camera import _safe_source_label  # noqa: E402
from config import cfg  # noqa: E402

BIN = 0.1          # s: resolución de las series
MAX_LAG = 4.0      # s: se busca el desfase entre -4 y +4 s


def sub_url() -> str:
    main = os.getenv("CAMERA_SOURCE", "")
    return cfg.detect_source or (main.replace("subtype=0", "subtype=1") if "subtype=0" in main else "")


def registrar(url: str, nombre: str, fin: float, salida: list, estado: dict) -> None:
    import av
    try:
        c = av.open(url, options={"rtsp_transport": "tcp", "fflags": "nobuffer"}, timeout=8)
    except Exception as e:  # noqa: BLE001
        estado[nombre] = f"no abrió: {type(e).__name__}"
        return
    s = c.streams.video[0]
    estado[nombre] = f"{s.codec_context.width}x{s.codec_context.height} {s.codec_context.name}"
    try:
        for p in c.demux(s):
            t = time.monotonic()
            if t > fin:
                break
            if p.size:
                salida.append((t, p.size, bool(p.is_keyframe)))
    except Exception as e:  # noqa: BLE001
        estado[nombre + "_error"] = f"{type(e).__name__}: {e}"
    finally:
        try:
            c.close()
        except Exception:  # noqa: BLE001
            pass


def serie(paquetes: list, t0: float, n: int) -> np.ndarray:
    """Bytes de paquetes NO clave por intervalo, sin la tendencia lenta y normalizada."""
    v = np.zeros(n)
    for t, size, clave in paquetes:
        if clave:
            continue  # los keyframes son periódicos y no dependen del movimiento
        i = int((t - t0) / BIN)
        if 0 <= i < n:
            v[i] += size
    v = np.log1p(v)
    k = int(2.0 / BIN)
    tend = np.convolve(v, np.ones(k) / k, mode="same")
    v = v - tend
    sd = v.std()
    return v / sd if sd > 0 else v


def correlacion(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """corr[l] con l en pasos de BIN: positivo = el principal (b) llega DESPUÉS que el sub-stream (a)."""
    m = int(MAX_LAG / BIN)
    lags = np.arange(-m, m + 1)
    out = np.zeros(len(lags))
    for j, l in enumerate(lags):
        x, y = (a[: len(a) - l], b[l:]) if l >= 0 else (a[-l:], b[: len(b) + l])
        k = min(len(x), len(y))
        out[j] = float(np.mean(x[:k] * y[:k])) if k > 10 else 0.0
    return lags * BIN, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segundos", type=int, default=300)
    seg = ap.parse_args().segundos

    urls = {"principal": os.getenv("CAMERA_SOURCE", ""), "sub": sub_url()}
    if not all(urls.values()):
        print("Falta CAMERA_SOURCE o el sub-stream en el .env del agente.")
        return 2
    print(f"Midiendo {seg} s. Principal: {_safe_source_label(urls['principal'])} | "
          f"Sub-stream: {_safe_source_label(urls['sub'])}", flush=True)

    t0 = time.monotonic()
    fin = t0 + seg
    datos: dict = {"principal": [], "sub": []}
    estado: dict = {}
    hilos = [threading.Thread(target=registrar, args=(urls[k], k, fin, datos[k], estado), daemon=True) for k in urls]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join(seg + 30)

    print("Streams:", estado, flush=True)
    if min(len(datos["principal"]), len(datos["sub"])) < 100:
        print("Muy pocos paquetes: la cámara no respondió bien. No se puede medir.")
        return 3
    n = int(seg / BIN)
    a, b = serie(datos["sub"], t0, n), serie(datos["principal"], t0, n)
    lags, c = correlacion(a, b)
    i = int(np.argmax(c))
    # ¿el pico se destaca del ruido? (mejor pico fuera de ±0,5 s del máximo)
    fuera = np.abs(lags - lags[i]) > 0.5
    ruido = float(np.max(c[fuera])) if fuera.any() else 0.0
    # estabilidad: mitad contra mitad
    mitad = n // 2
    mitades = []
    for sl in (slice(0, mitad), slice(mitad, n)):
        lg, cc = correlacion(a[sl], b[sl])
        mitades.append(round(float(lg[int(np.argmax(cc))]), 2))
    actividad = float(np.mean(np.abs(a) > 1.5))
    res = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "segundos": seg,
        "paquetes": {k: len(v) for k, v in datos.items()},
        "desfase_sec": round(float(lags[i]), 2),
        "corr_pico": round(float(c[i]), 3),
        "corr_mejor_ruido": round(ruido, 3),
        "corr_en_cero": round(float(c[int(MAX_LAG / BIN)]), 3),
        "desfase_por_mitad": mitades,
        "fraccion_con_movimiento": round(actividad, 3),
        "interpretacion": "positivo = el principal llega DESPUÉS que el sub-stream; es el valor de MAIN_STREAM_OFFSET_SEC",
        "offset_configurado": cfg.main_stream_offset_sec,
        "tolerancia_configurada": cfg.main_match_max_sec,
    }
    confiable = c[i] > 0.15 and c[i] > 1.5 * max(ruido, 0.01) and abs(mitades[0] - mitades[1]) <= 0.3
    res["confiable"] = bool(confiable)
    out = AGENT / "audit_runs" / f"desfase_{datetime.now():%Y%m%d_%H%M}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
