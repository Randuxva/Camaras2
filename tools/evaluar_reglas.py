"""
Evalúa reglas del portón sobre trayectorias reales (tools/extraer_trayectorias.py): pasa los mismos cuadros por el
motor ROI con distintas configuraciones y guarda cada aviso. Solo lee; no usa red ni nube.

Uso (desde anpr-edge/):
    venv/Scripts/python.exe tools/evaluar_reglas.py <trayectorias.jsonl> <salida.json>

Variantes: "actual" (todas las rutas de rescate), "estricta_sin_union" y "estricta" (cruce + unión de
seguimientos cortados). Cada aviso guarda tipo, ruta, si se archivaría (sin cruce), hora del video y posición.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

AGENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.disable(logging.INFO)

from config import cfg  # noqa: E402
from core.roi_engine import ROIEngine  # noqa: E402

VARIANTES = {
    "actual": dict(solo_cruces=False),
    "estricta_sin_union": dict(solo_cruces=True, union_sec=0.0),
    "estricta": dict(solo_cruces=True, union_sec=cfg.roi_union_sec),
}


def correr(cuadros: list, **kw) -> tuple[list, dict]:
    eng = ROIEngine(line_a_y_ratio=cfg.roi_line_a_y_ratio, line_b_y_ratio=cfg.roi_line_b_y_ratio,
                    direction_timeout_sec=cfg.roi_direction_timeout_sec, track_lost_sec=cfg.roi_track_lost_sec,
                    min_displacement_ratio=cfg.roi_min_displacement_ratio, continuation_sec=cfg.roi_continuation_sec,
                    event_hold_sec=cfg.roi_event_hold_sec, **kw)
    avisos = []

    def guardar(evs, t):
        for e in evs:
            x1, y1, x2, y2 = e.bbox
            avisos.append({"t": round(e.timestamp, 2), "t_emitido": round(t, 2), "tipo": e.event_type.value,
                           "ruta": e.reason, "archivado": e.sin_movimiento, "track": e.track_id,
                           "n": e.n_points, "y": round(e.centroid_y, 3), "caja": [x1, y1, x2, y2]})

    t = 0.0
    for c in cuadros:
        t = c["t"]
        for tid, _cl, x1, y1, x2, y2, _cf in c["det"]:
            eng.update(tid, (x1, y1, x2, y2), c["h"], c["w"], timestamp=t)
        guardar(eng.collect_finished(t), t)
    for extra in (t + 30, t + 60):
        guardar(eng.collect_finished(extra), extra)
    resumen = {"avisos": sum(1 for a in avisos if not a["archivado"]), "archivados": sum(1 for a in avisos if a["archivado"]),
               "unidos": eng.total_unidos, "continuaciones": eng.total_continuations}
    return avisos, resumen


def main() -> int:
    cuadros = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    salida = {}
    for nombre, kw in VARIANTES.items():
        avisos, res = correr(cuadros, **kw)
        salida[nombre] = {"resumen": res, "avisos": avisos}
        print(f"{nombre:20s} {res}")
    json.dump(salida, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
