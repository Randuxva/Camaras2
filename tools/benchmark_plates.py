"""
Mide la lectura local de patentes contra un set etiquetado (recortes reales del portón).

Uso (desde anpr-edge/):
  venv/Scripts/python.exe tools/benchmark_plates.py <dir> [modelo_ocr ...]

<dir> contiene crops/<evento>_<i>.jpg (como los guarda tools/audit_run.py) y labels.json:
  {"<timestamp>_t<track>": "AA123BB" | ["AA123BB", "AA128BB"] | null}
null = la patente no se lee en ningún recorte (lo correcto es NO aceptar nada).

Métricas por evento (vehículo):
  aceptadas_ok   lectura local "fuerte" y correcta (se aceptaría sola)
  aceptadas_mal  lectura local "fuerte" e INCORRECTA (el error grave: patente equivocada en la base)
  candidata_ok   no fuerte, pero el voto acierta (sirve como segunda opinión para Gemini)
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.plate_reader import PlateReader, fuse_reads  # noqa: E402

DETECTOR = "yolo-v9-t-384-license-plate-end2end"


def load(dir_: Path):
    labels = json.loads((dir_ / "labels.json").read_text(encoding="utf-8"))
    events = defaultdict(list)
    for f in sorted((dir_ / "crops").glob("*.jpg")):
        m = re.match(r"(\d+_t\d+)_\w+?_(\d+)\.jpg$", f.name)
        if m and m.group(1) in labels:
            events[m.group(1)].append(f)
    return labels, events


def run(dir_: Path, ocr_model: str) -> dict:
    labels, events = load(dir_)
    reader = PlateReader(DETECTOR, ocr_model)
    stats = defaultdict(int)
    t_total, n_crops = 0.0, 0
    for key, files in sorted(events.items()):
        truth = labels[key]
        ok_set = set(truth) if isinstance(truth, list) else ({truth} if truth else set())
        crops = [cv2.imread(str(f)) for f in files]
        t0 = time.perf_counter()
        reads = reader.read_many(crops)
        t_total += time.perf_counter() - t0
        n_crops += len(crops)
        fused = fuse_reads(reads)
        strong = fused.is_strong()
        correct = fused.plate in ok_set
        if strong and correct:
            stats["aceptadas_ok"] += 1
        elif strong and not correct:
            stats["aceptadas_mal"] += 1
        elif correct:
            stats["candidata_ok"] += 1
        stats["legibles"] += 1 if ok_set else 0
        print(f"  {key:16} verdad={'/'.join(sorted(ok_set)) or '-':17} voto={fused.plate or '-':8} "
              f"apoyo={fused.support}/{fused.n_reads} min={fused.min_char_conf:.2f} "
              f"{'FUERTE' if strong else '      '} {'OK' if correct else ('--' if not ok_set and not fused.plate else 'X')}"
              f"  lecturas={[r.text + f'({r.mean_prob:.2f})' for r in reads]}")
    stats["eventos"] = len(events)
    stats["ms_por_recorte"] = round(1000 * t_total / max(1, n_crops))
    return dict(stats)


if __name__ == "__main__":
    d = Path(sys.argv[1])
    models = sys.argv[2:] or ["cct-s-v2-global-model"]
    for m in models:
        print(f"\n== {m}")
        print(json.dumps(run(d, m), ensure_ascii=False))
