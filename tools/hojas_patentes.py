"""
Hojas de control visual: cada evento de la cámara en una celda con la foto de la patente, la foto
del vehículo y lo que registró el sistema. Sirve para verificar a ojo TODOS los vehículos.

    python tools/hojas_patentes.py            → audit_runs/semana/hojas/hoja_NN.jpg + indice.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

AGENT = Path(__file__).resolve().parents[1]
D = AGENT / "audit_runs" / "semana"
FOTOS = D / "fotos"
OUT = D / "hojas"
OUT.mkdir(exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

eventos = list(csv.DictReader(open(D / "eventos.csv", encoding="utf-8-sig")))
libro = {e["id"]: e for e in json.load(open(D / "libro.json", encoding="utf-8"))["libro"]}
COLS, FILAS = 4, 10
W_PAT, W_VEH, H = 330, 190, 120


def cargar(nombre, w, h):
    p = FOTOS / nombre
    img = cv2.imread(str(p)) if p.exists() else None
    if img is None:
        return np.full((h, w, 3), 40, np.uint8)
    s = min(w / img.shape[1], h / img.shape[0])
    img = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
    lienzo = np.full((h, w, 3), 25, np.uint8)
    y, x = (h - img.shape[0]) // 2, (w - img.shape[1]) // 2
    lienzo[y:y + img.shape[0], x:x + img.shape[1]] = img
    return lienzo


celdas, indice = [], []
for i, ev in enumerate(e for e in eventos if e["subida"] != "FALLIDO"):
    e = libro[ev["event_id"]]
    # Mismo nombre que arma el agente al guardar el recorte (cloud/uploader.py)
    base = f"crop_vehicle_{e['track_id']}_{e['timestamp']}_{e['event_type'].lower()}.webp"
    raiz = base.rsplit(".", 1)[0]
    pat = cargar(raiz + "_plate0.webp", W_PAT, H - 22)
    vehi = cargar(base, W_VEH, H - 22)
    celda = np.full((H, W_PAT + W_VEH, 3), 15, np.uint8)
    celda[22:, :W_PAT] = pat
    celda[22:, W_PAT:] = vehi
    final = ev["patente_final"]
    reg = "REVISAR" if (not final or final.startswith("DESC-")) else final
    color = (0, 200, 255) if reg == "REVISAR" else (120, 255, 120)
    texto = f"{i + 1} {ev['direccion'][0]} {ev['hora'][:11]} -> {reg}"
    cv2.putText(celda, texto, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.rectangle(celda, (0, 0), (celda.shape[1] - 1, H - 1), (70, 70, 70), 1)
    celdas.append(celda)
    indice.append({"n": i + 1, "event_id": ev["event_id"], "hora": ev["hora"], "direccion": ev["direccion"], "registrado": reg,
                   "portón": ev["patente_portón"], "gemini": ev["patente_gemini"], "decision": ev["decision"], "hoja": i // (COLS * FILAS) + 1})

por_hoja = COLS * FILAS
for h in range(0, len(celdas), por_hoja):
    grupo = celdas[h:h + por_hoja]
    while len(grupo) % COLS:
        grupo.append(np.zeros_like(celdas[0]))
    filas = [np.hstack(grupo[k:k + COLS]) for k in range(0, len(grupo), COLS)]
    cv2.imwrite(str(OUT / f"hoja_{h // por_hoja + 1:02d}.jpg"), np.vstack(filas), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
with open(OUT / "indice.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(indice[0]))
    w.writeheader()
    w.writerows(indice)
print(f"{len(celdas)} eventos en {(len(celdas) + por_hoja - 1) // por_hoja} hojas → {OUT}")
