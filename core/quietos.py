"""
Salvaguardas para los objetos quietos (ROI_SOLO_MOVIMIENTO).

Un objeto "quieto" es un track que no cruzó el portón ni se movió claramente y se perdió de vista (por
ejemplo un auto estacionado afuera, visto a través de la reja, que el seguimiento pierde y re-detecta
una y otra vez). NUNCA se descarta uno por no tener patente: solo se deja de mandar a "En revisión" el que
cumple TODO esto a la vez:

  1. no se le encontró ninguna patente (ni en la segunda lectura con la imagen realzada),
  2. no es una moto,
  3. su posición en el cuadro YA tuvo varios avisos iguales sin patente en las últimas 24 h
     (un lugar recurrente: el auto estacionado). Un lugar nuevo se envía siempre.

Nada se borra: todo lo suprimido queda con su foto en la carpeta `quietos/` y en la tabla `quietos` de
`quietos.db`, sin borrado automático.

Modos (ROI_SOLO_MOVIMIENTO):  "false" = comportamiento anterior · "sombra" = marca lo que suprimiría
pero lo ENVÍA TODO · "true" = suprime de verdad lo que cumple las tres condiciones.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)

Caja = Tuple[float, float, float, float]      # x1, y1, x2, y2 normalizados (0..1)

VENTANA_SEG = 24 * 3600
MIN_REPETICIONES = 5          # avisos iguales previos para considerar un lugar "recurrente"
IOU_LUGAR = 0.70              # cuánto debe coincidir la caja para ser "el mismo lugar"
CONTENCION_LUGAR = 0.85       # o cuánto de la caja más chica debe quedar dentro de la otra

ENVIAR_NORMAL = "ENVIAR_NORMAL"                  # no es un objeto quieto
ENVIAR_PATENTE = "ENVIAR_CON_PATENTE"            # quieto, pero tiene patente: un auto real parado
ENVIAR_MOTO = "ENVIAR_MOTO"                      # quieto sin patente, pero es una moto
ENVIAR_NUEVO = "ENVIAR_LUGAR_NUEVO"              # quieto sin patente en un lugar que casi no se repite
ENVIAR_SOMBRA = "ENVIAR_SOMBRA_HABRIA_SUPRIMIDO"  # modo sombra: se envía igual
SUPRIMIR = "SUPRIMIR_LUGAR_RECURRENTE"
# Regla estricta del portón (ROI_SOLO_CRUCES): lo que no cruzó la línea no va al panel; se archiva con su foto.
ARCHIVAR_SIN_CRUCE = "ARCHIVADO_SIN_CRUCE"
SOMBRA_SIN_CRUCE = "ENVIAR_SOMBRA_SIN_CRUCE"        # modo sombra de la regla estricta: se envía igual


def iou(a: Caja, b: Caja) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def mismo_lugar(a: Caja, b: Caja) -> bool:
    """
    ¿Dos cajas son "el mismo lugar" del cuadro? El auto estacionado afuera se ve a veces entero y a veces como una
    franja angosta entre los barrotes, así que además del solapamiento normal (IoU) vale que una caja quede
    contenida en la otra.
    """
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    menor = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    contenida = inter / menor if menor > 0 else 0.0
    return iou(a, b) >= IOU_LUGAR or contenida >= CONTENCION_LUGAR


def realzar(crop: np.ndarray) -> np.ndarray:
    """Segunda oportunidad de lectura: amplía 2x y mejora el contraste local (poca luz, desenfoque leve)."""
    if crop is None or crop.size == 0:
        return crop
    grande = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(grande, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


class RegistroQuietos:
    def __init__(self, db_path: str, carpeta: str, modo: str = "false",
                 min_repeticiones: int = MIN_REPETICIONES, ventana_seg: int = VENTANA_SEG,
                 estricto: bool = False) -> None:
        self.modo = modo
        self.estricto = estricto          # regla estricta: todo lo que no cruzó se archiva, sin excepciones
        self.min_rep = min_repeticiones
        self.ventana = ventana_seg
        self.db_path = db_path
        self.carpeta = Path(carpeta)
        self._lock = threading.Lock()
        self._cache: list = []            # (t, caja) de avisos quietos sin patente recientes
        self.enviados = 0
        self.suprimidos = 0
        self.sombra = 0
        self._ultimo_resumen = time.time()
        self._init_db()
        self._cargar_cache()

    # ── base ────────────────────────────────────────────────────────────────
    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=10.0)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._lock, self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS quietos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL, track_id INTEGER, tipo TEXT, ruta TEXT, n_puntos INTEGER,
                    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                    clase TEXT, con_patente INTEGER, segunda_lectura INTEGER,
                    repeticiones_previas INTEGER, decision TEXT, foto TEXT
                )""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_quietos_ts ON quietos(ts)")

    def _cargar_cache(self) -> None:
        try:
            with self._lock, self._conn() as c:
                filas = c.execute(
                    "SELECT ts, x1, y1, x2, y2 FROM quietos WHERE con_patente = 0 AND ts >= ?",
                    (time.time() - self.ventana,)).fetchall()
            self._cache = [(f["ts"], (f["x1"], f["y1"], f["x2"], f["y2"])) for f in filas if f["x1"] is not None]
        except Exception as e:  # nunca debe impedir el arranque
            log.warning(f"No se pudo cargar el historial de quietos: {e}")

    # ── decisión ────────────────────────────────────────────────────────────
    def repeticiones(self, caja: Optional[Caja], ahora: float) -> int:
        if caja is None:
            return 0
        corte = ahora - self.ventana
        self._cache = [(t, b) for t, b in self._cache if t >= corte]
        return sum(1 for _, b in self._cache if mismo_lugar(caja, b))

    def decidir(self, *, sin_movimiento: bool, con_patente: bool, clase: str, caja: Optional[Caja],
                ahora: Optional[float] = None) -> Tuple[str, int]:
        """Devuelve (decisión, repeticiones previas en ese lugar). Solo SUPRIMIR no envía."""
        ahora = ahora if ahora is not None else time.time()
        if not sin_movimiento:
            return ENVIAR_NORMAL, 0
        if self.estricto:
            return (SOMBRA_SIN_CRUCE if self.modo == "sombra" else ARCHIVAR_SIN_CRUCE), 0
        if con_patente:
            return ENVIAR_PATENTE, 0
        rep = self.repeticiones(caja, ahora)
        if clase == "motorcycle":
            return ENVIAR_MOTO, rep
        if rep < self.min_rep or caja is None:
            return ENVIAR_NUEVO, rep
        return (ENVIAR_SOMBRA if self.modo == "sombra" else SUPRIMIR), rep

    # ── constancia (nunca se borra) ─────────────────────────────────────────
    def registrar(self, *, track_id: int, tipo: str, ruta: str, n_puntos: int, caja: Optional[Caja], clase: str,
                  con_patente: bool, segunda_lectura: bool, repeticiones: int, decision: str,
                  crops: Sequence[np.ndarray] = (), ahora: Optional[float] = None) -> None:
        ahora = ahora if ahora is not None else time.time()
        foto = ""
        if decision in (SUPRIMIR, ENVIAR_SOMBRA, ARCHIVAR_SIN_CRUCE, SOMBRA_SIN_CRUCE) and crops:
            try:
                self.carpeta.mkdir(parents=True, exist_ok=True)
                foto = str(self.carpeta / f"quieto_{int(ahora)}_t{track_id}.webp")
                cv2.imwrite(foto, crops[0], [int(cv2.IMWRITE_WEBP_QUALITY), 85])
            except Exception as e:
                log.warning(f"No se pudo archivar la foto del quieto: {e}")
                foto = ""
        c = caja or (None, None, None, None)
        try:
            with self._lock, self._conn() as db:
                db.execute(
                    "INSERT INTO quietos (ts, track_id, tipo, ruta, n_puntos, x1, y1, x2, y2, clase, con_patente, "
                    "segunda_lectura, repeticiones_previas, decision, foto) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (ahora, track_id, tipo, ruta, n_puntos, c[0], c[1], c[2], c[3], clase, int(con_patente),
                     int(segunda_lectura), repeticiones, decision, foto))
        except Exception as e:
            log.warning(f"No se pudo registrar el quieto: {e}")
        if not con_patente and caja is not None:
            self._cache.append((ahora, caja))
        if decision in (SUPRIMIR, ARCHIVAR_SIN_CRUCE):
            self.suprimidos += 1
        elif decision in (ENVIAR_SOMBRA, SOMBRA_SIN_CRUCE):
            self.sombra += 1
        elif decision != ENVIAR_NORMAL:
            self.enviados += 1
        if ahora - self._ultimo_resumen > 3600 and (self.suprimidos or self.sombra or self.enviados):
            self._ultimo_resumen = ahora
            log.info(f"Objetos quietos (última hora): enviados {self.enviados}, "
                     f"suprimidos {self.suprimidos}, en sombra {self.sombra}.")
            self.enviados = self.suprimidos = self.sombra = 0
