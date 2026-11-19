"""
Agrupa los avisos REPETIDOS de un mismo vehículo (ROI_AGRUPAR_REPETIDOS).

Un auto que se queda parado o maniobrando frente al portón se pierde de vista y se re-detecta una y otra
vez, y cada vez genera un aviso nuevo (una furgoneta llegó a más de 40). Un aviso NO se envía como nuevo
solo si se cumplen TODAS estas condiciones:

  1. su patente se leyó con firmeza (la lectura local con 2 o más cuadros),
  2. esa misma patente ya tuvo un aviso hace menos de 10 minutos,
  3. el sentido es el mismo que el del aviso anterior (entrada tras entrada, salida tras salida, sin
     dirección tras sin dirección). Un sentido contrario, o pasar de "sin dirección" a una dirección, o al
     revés, SIEMPRE se envía: una entrada, una salida y otra entrada son tres pasos distintos,
  4. no pasaron más de 60 minutos desde el último aviso que sí se envió (así un auto que se queda
     horas sigue dejando constancia cada hora),
  5. si el sentido es entrada o salida, el aviso anterior YA TERMINÓ BIEN en la nube (se registró la
     entrada o la salida, o se ignoró como repetido) y pasaron al menos 15 s para que esa respuesta haya
     llegado. Si el anterior fue a revisión o dio error, el siguiente SIEMPRE se envía: puede ser el que sí
     se registre. (Un aviso "sin dirección" tras otro "sin dirección" sí se agrupa: la nube los trata igual.)

Nada se borra: el aviso agrupado queda con su foto en la carpeta `repetidos/` y en la tabla `repetidos` de
`quietos.db`, sin borrado automático.

Modos (ROI_AGRUPAR_REPETIDOS): "false" = comportamiento anterior · "sombra" = marca lo que agruparía pero lo
ENVÍA TODO · "true" = agrupa de verdad.
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

VENTANA_SEG = 10 * 60
TOPE_SEG = 60 * 60

ENVIAR_SIN_PATENTE_FIRME = "ENVIAR_SIN_PATENTE_FIRME"
ENVIAR_PRIMERO = "ENVIAR_PRIMER_AVISO"
ENVIAR_OTRO_SENTIDO = "ENVIAR_OTRO_SENTIDO"
ENVIAR_TOPE = "ENVIAR_CONSTANCIA_DE_LA_HORA"
ENVIAR_SIN_CONFIRMAR = "ENVIAR_ANTERIOR_SIN_CONFIRMAR"      # el aviso anterior no terminó bien (o no hay respuesta aún)
RESULTADOS_OK = frozenset({"CREATE_ENTRY", "EXIT", "REPLACE_STALE_ENTRY", "EXIT_ABONADO_SIN_ENTRADA",
                           "DEBOUNCE_IGNORE", "DUPLICATE_IGNORED", "EXIT_ABANDONADO", "LATE_ENTRY_CLOSED"})
ESPERA_RESPUESTA_SEG = 15
ENVIAR_SOMBRA = "ENVIAR_SOMBRA_HABRIA_AGRUPADO"
AGRUPAR = "AGRUPAR_REPETIDO"


class RegistroRepetidos:
    def __init__(self, db_path: str, carpeta: str, modo: str = "false",
                 ventana_seg: int = VENTANA_SEG, tope_seg: int = TOPE_SEG) -> None:
        self.modo = modo
        self.ventana = ventana_seg
        self.tope = tope_seg
        self.db_path = db_path
        self.carpeta = Path(carpeta)
        self._lock = threading.Lock()
        self._ultimo: dict = {}          # patente -> [ts último aviso, dirección, ts último enviado]
        self.agrupados = 0
        self.sombra = 0
        self._init_db()
        self._cargar()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=10.0)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._lock, self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS repetidos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL, patente TEXT, direccion TEXT, track_id INTEGER, ruta TEXT,
                    decision TEXT, segundos_desde_el_anterior REAL, foto TEXT
                )""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_repetidos_ts ON repetidos(ts)")

    def _cargar(self) -> None:
        """Recupera el último aviso de cada patente tras un reinicio (solo lo de la última hora)."""
        try:
            with self._lock, self._conn() as c:
                filas = c.execute("SELECT ts, patente, direccion, decision FROM repetidos WHERE ts >= ? ORDER BY ts",
                                  (time.time() - self.tope,)).fetchall()
            for f in filas:
                enviado = not (f["decision"] in (AGRUPAR, ENVIAR_SOMBRA))
                prev = self._ultimo.get(f["patente"])
                self._ultimo[f["patente"]] = [f["ts"], f["direccion"], f["ts"] if enviado else (prev[2] if prev else f["ts"])]
        except Exception as e:
            log.warning(f"No se pudo cargar el historial de repetidos: {e}")

    def decidir(self, *, patente: Optional[str], firme: bool, direccion: str, ahora: Optional[float] = None,
                resultado_previo=None) -> Tuple[str, Optional[float]]:
        """
        Devuelve (decisión, segundos desde el aviso anterior de esa patente). Solo AGRUPAR no envía.
        resultado_previo(patente, direccion, desde, hasta) -> (acción, timestamp) | None: cómo terminó en la nube
        el aviso anterior de esa patente y ese sentido (OfflineQueue.accion_previa).
        """
        ahora = ahora if ahora is not None else time.time()
        if not patente or not firme:
            return ENVIAR_SIN_PATENTE_FIRME, None
        prev = self._ultimo.get(patente)
        if prev is None or ahora - prev[0] > self.ventana:
            return ENVIAR_PRIMERO, None
        dt = ahora - prev[0]
        if direccion != prev[1]:
            return ENVIAR_OTRO_SENTIDO, dt
        if ahora - prev[2] > self.tope:
            return ENVIAR_TOPE, dt
        if direccion != "DESCONOCIDA":
            # Una entrada o una salida solo se agrupa si el aviso anterior ya terminó bien en la nube
            previo = None
            try:
                previo = resultado_previo(patente, direccion, ahora - self.ventana, ahora - ESPERA_RESPUESTA_SEG) if resultado_previo else None
            except Exception as e:
                log.warning(f"No se pudo consultar el resultado previo ({e}): se envía.")
            if not previo or previo[0] not in RESULTADOS_OK:
                return ENVIAR_SIN_CONFIRMAR, dt
        return (ENVIAR_SOMBRA if self.modo == "sombra" else AGRUPAR), dt

    def registrar(self, *, patente: Optional[str], firme: bool, direccion: str, track_id: int, ruta: str,
                  decision: str, desde_anterior: Optional[float], crops: Sequence[np.ndarray] = (),
                  ahora: Optional[float] = None) -> None:
        ahora = ahora if ahora is not None else time.time()
        if not patente or not firme:
            return
        foto = ""
        if decision in (AGRUPAR, ENVIAR_SOMBRA) and crops:
            try:
                self.carpeta.mkdir(parents=True, exist_ok=True)
                foto = str(self.carpeta / f"repetido_{int(ahora)}_t{track_id}.webp")
                cv2.imwrite(foto, crops[0], [int(cv2.IMWRITE_WEBP_QUALITY), 85])
            except Exception as e:
                log.warning(f"No se pudo archivar la foto del repetido: {e}")
                foto = ""
        try:
            with self._lock, self._conn() as db:
                db.execute("INSERT INTO repetidos (ts, patente, direccion, track_id, ruta, decision, "
                           "segundos_desde_el_anterior, foto) VALUES (?,?,?,?,?,?,?,?)",
                           (ahora, patente, direccion, track_id, ruta, decision, desde_anterior, foto))
        except Exception as e:
            log.warning(f"No se pudo registrar el repetido: {e}")
        enviado = decision not in (AGRUPAR, ENVIAR_SOMBRA)
        prev = self._ultimo.get(patente)
        self._ultimo[patente] = [ahora, direccion, ahora if (enviado or prev is None) else prev[2]]
        if decision == AGRUPAR:
            self.agrupados += 1
        elif decision == ENVIAR_SOMBRA:
            self.sombra += 1
