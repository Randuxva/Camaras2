"""
Motor ROI Bidireccional para puerta única de estacionamiento (2026).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GEOMETRÍA (cámara DENTRO del garaje, apuntando hacia la calle):

  y=0  ┌────────────────────────┐
       │  CALLE                 │  ← tráfico que pasa / autos estacionados enfrente
  y=B  │ ─ ─ LÍNEA B ─ ─ ─ ─ ─ │  ← borde calle / portón
       │  PORTAL (vereda)       │  ← el auto de frente: mejor foto de patente
  y=A  │ ━━━ LÍNEA A ━━━━━━━━━ │  ← borde portón / rampa
       │  INTERIOR (rampa)      │
  y=h  └────────────────────────┘  ← cámara

Posición del vehículo = borde inferior del bbox (punto de apoyo en el piso).

POR QUÉ ESTE DISEÑO (auditoría 2026-09-28):
  La versión anterior exigía ver el cruce B↓→A↓ (o A↑→B↑) en frames procesados.
  Con ~0,5 inferencias/s y autos que cruzan el portón en 2–4 s, casi nunca se
  observaban ambos cruces; el disparo de respaldo marcaba todo como "ENTRADA".
  Resultado: 1 de 11 vehículos capturado y con dirección invertida.

REGLAS:
  1. Dirección decidida cuando la trayectoria cruza el portal completo:
       primera zona CALLE    y ahora INTERIOR  → ENTRADA
       primera zona INTERIOR y ahora CALLE     → SALIDA
     El evento se emite `event_hold_sec` después (o al perder el track) para
     seguir juntando fotos del vehículo más cerca de la cámara (patente más grande).
  2. Evento diferido cuando el track se pierde (no visto por `track_lost_sec`)
     y llegó a tocar PORTAL/INTERIOR:
       desplazamiento vertical  > +umbral → ENTRADA (se acercó a la cámara)
       desplazamiento vertical  < -umbral → SALIDA  (se alejó de la cámara)
       sin desplazamiento claro          → por zona inicial (INTERIOR ⇒ SALIDA,
                                            CALLE ⇒ ENTRADA), si no DESCONOCIDA
  3. Se ignoran: tracks que nunca salieron de la CALLE (tráfico/estacionados) y
     movimientos puramente horizontales que no llegan al INTERIOR.
  4. Continuación (cambio de ID del tracker): cuando un auto que entra pasa pegado
     a la cámara, YOLO/BoT-SORT suele asignarle un ID nuevo. Un track que NACE en el
     INTERIOR hasta `continuation_sec` después de una ENTRADA es el mismo vehículo
     (idem un track que nace más lejos que el último punto de una SALIDA).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


# ── Enumeraciones ────────────────────────────────────────────────────────────

class Zone(Enum):
    CALLE = "CALLE"
    PORTAL = "PORTAL"
    INTERIOR = "INTERIOR"


class GateEvent(Enum):
    ENTRADA = "ENTRADA"
    SALIDA = "SALIDA"
    DESCONOCIDA = "DESCONOCIDA"


# ── Estructuras de datos ─────────────────────────────────────────────────────

@dataclass
class TrackPoint:
    t: float          # time.monotonic()
    y: float          # borde inferior del bbox, normalizado 0..1
    x: float          # centro horizontal, normalizado 0..1
    zone: Zone


@dataclass
class TrackRecord:
    track_id: int
    points: List[TrackPoint] = field(default_factory=list)
    event_triggered: bool = False
    last_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    last_norm: Optional[Tuple[float, float, float, float]] = None   # caja 0..1 (x1, y1, x2, y2)
    pending: Optional[Tuple["GateEvent", str, float]] = None  # (dirección, motivo, t decisión)

    @property
    def first(self) -> TrackPoint:
        return self.points[0]

    @property
    def last(self) -> TrackPoint:
        return self.points[-1]

    @property
    def last_seen_ts(self) -> float:
        return self.points[-1].t if self.points else 0.0

    @property
    def max_y(self) -> float:
        return max(p.y for p in self.points)

    @property
    def min_y(self) -> float:
        return min(p.y for p in self.points)


@dataclass
class ROIEvent:
    """Evento de paso por el portal."""
    track_id: int
    event_type: GateEvent
    timestamp: float                  # time.monotonic() del último punto
    centroid_y: float                 # y normalizado del último punto
    bbox: Tuple[int, int, int, int]   # último bbox observado
    reason: str                       # CRUCE_COMPLETO | TRACK_PERDIDO_*
    n_points: int
    # True: el objeto no cruzó ni se movió claramente (solo con solo_movimiento). El consumidor lo envía
    # únicamente si el lector local encontró una patente; sin patente lo descarta (objeto quieto).
    sin_movimiento: bool = False
    norm_box: Optional[Tuple[float, float, float, float]] = None   # caja 0..1 del último punto (lugar en el cuadro)


# ── Motor Principal ──────────────────────────────────────────────────────────

class ROIEngine:
    """
    Gestiona el estado de todos los tracks visibles y emite ROIEvent cuando un
    vehículo pasa por el portal (ENTRADA / SALIDA / DESCONOCIDA).

    Todos los tiempos deben ser time.monotonic(). Pensado para un único hilo.
    """

    def __init__(
        self,
        line_a_y_ratio: float = 0.68,
        line_b_y_ratio: float = 0.56,
        direction_timeout_sec: float = 8.0,
        polygon_ratios: str = "",
        track_lost_sec: float = 2.0,
        min_displacement_ratio: float = 0.04,
        continuation_sec: float = 8.0,
        event_hold_sec: float = 1.5,
        solo_movimiento: bool = False,
        solo_cruces: bool = False,
        union_sec: float = 3.0,
    ) -> None:
        assert line_b_y_ratio < line_a_y_ratio, (
            f"Line B ({line_b_y_ratio}) debe ser más alta (y menor) que Line A ({line_a_y_ratio})"
        )
        self.line_a_ratio = line_a_y_ratio
        self.line_b_ratio = line_b_y_ratio
        self.timeout_sec = direction_timeout_sec
        self.track_lost_sec = track_lost_sec
        self.min_disp = min_displacement_ratio
        self.continuation_sec = continuation_sec
        self.event_hold_sec = event_hold_sec
        # True: un objeto que se pierde SIN haberse movido claramente no genera evento (solo cruces y
        # desplazamientos). Un objeto quieto (p. ej. un auto estacionado visto a través de la reja) se
        # pierde y se re-detecta una y otra vez; con la ruta de rescate cada vez avisaba "sin dirección".
        self.solo_movimiento = solo_movimiento
        # REGLA ESTRICTA (solo_cruces): el único aviso es el cruce. ENTRADA = el vehículo estuvo arriba de la
        # línea B (calle) y llegó a la línea A; SALIDA = estuvo abajo de la línea A (interior) y llegó a la B.
        # Un auto de arriba que solo toca la línea B (estacionado detrás de la reja, esperando en la puerta)
        # no genera nada. Lo que se pierde de vista sin cruzar no se avisa: sale marcado (sin_movimiento) para
        # que el consumidor lo ARCHIVE con su foto en el equipo (nunca se borra), no para mandarlo al panel.
        # Para no perder vehículos reales cuyo seguimiento se corta a mitad del portón (cambio de ID del
        # tracker, reja, faros), un track perdido sin cruzar espera `union_sec`: si en ese lapso nace otro
        # track pegado a donde se perdió, siguiendo el mismo sentido, se UNEN y el cruce se evalúa sobre la
        # trayectoria completa del vehículo (la regla sigue siendo la misma: calle → línea A, interior → línea B).
        self.solo_cruces = solo_cruces
        self.union_sec = union_sec
        self._perdidos: list = []                    # (t perdido, TrackRecord) esperando una posible unión
        self.total_unidos: int = 0
        self.total_sin_cruce: int = 0
        self._last_event: Optional[Tuple[float, GateEvent, float, float]] = None  # (t, dirección, y, x finales)

        # Polígono ROI opcional (coords en ratio 0..1)
        self._polygon: Optional[np.ndarray] = self._parse_polygon(polygon_ratios)

        self._tracks: Dict[int, TrackRecord] = {}

        # Estadísticas de sesión
        self.total_entries: int = 0
        self.total_exits: int = 0
        self.total_unknown: int = 0
        self.total_dropped: int = 0
        self.total_quietos: int = 0              # perdidos sin movimiento claro (solo con solo_movimiento)
        self.total_continuations: int = 0
        self._confirmed: deque = deque(maxlen=10)    # (t, tipo, y, x) de ENTRADA/SALIDA recientes
        self._held_unknown: list = []                # (liberar_en, evento, x) DESCONOCIDA retenidos

    # ── API pública ─────────────────────────────────────────────────────────

    def zone_of(self, y_norm: float) -> Zone:
        if y_norm < self.line_b_ratio:
            return Zone.CALLE
        if y_norm < self.line_a_ratio:
            return Zone.PORTAL
        return Zone.INTERIOR

    def update(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        frame_height: int,
        frame_width: int,
        timestamp: Optional[float] = None,
    ) -> Optional[ROIEvent]:
        """
        Registra la posición actual del track. Los eventos se entregan en
        collect_finished(); se mantiene el retorno Optional por compatibilidad.
        """
        now = timestamp if timestamp is not None else time.monotonic()
        x1, y1, x2, y2 = bbox
        y = y2 / float(frame_height)
        x = (x1 + x2) / 2.0 / float(frame_width)

        if self._polygon is not None and not self._inside_polygon(x, y):
            return None

        rec = self._tracks.get(track_id)
        nuevo = rec is None
        if rec is None:
            rec = TrackRecord(track_id=track_id)
            self._tracks[track_id] = rec
        rec.points.append(TrackPoint(t=now, y=y, x=x, zone=self.zone_of(y)))
        rec.last_bbox = bbox
        rec.last_norm = (x1 / float(frame_width), y1 / float(frame_height), x2 / float(frame_width), y2 / float(frame_height))
        if len(rec.points) > 120:          # acotar memoria en tracks estáticos largos
            del rec.points[1:-100]         # conservar siempre el primer punto

        if rec.event_triggered or rec.pending:
            return None
        if len(rec.points) == 1 and self._is_continuation(rec):
            rec.event_triggered = True  # mismo vehículo que el evento anterior: no volver a emitir
            self.total_continuations += 1
            # Renovar la ventana: un auto que maniobra junto a la cámara reaparece varias veces
            _, kind, _, _ = self._last_event
            self._last_event = (now, kind, y, x)
            log.info("Track tratado como continuación del evento anterior (cambio de ID).",
                     extra={"track_id": track_id, "y_first": round(y, 3)})
            return None
        if nuevo and self.solo_cruces and self.union_sec > 0:
            self._unir_con_perdido(rec)

        kind: Optional[GateEvent] = self._cruce(rec)
        if kind is not None:
            rec.pending = (kind, "CRUCE_COMPLETO", now)
            self._last_event = (now, kind, y, x)  # ya cuenta para detectar cambios de ID
        return None

    # Fragmentos: un evento DESCONOCIDA se retiene este tiempo; si cerca (en tiempo y lugar) hay una
    # ENTRADA/SALIDA confirmada, es un pedazo del mismo vehículo (track partido) y se descarta.
    # Grabación 2026-10-02: una moto rápida y la salida de un auto generaron cada uno un evento
    # confirmado + uno DESCONOCIDA del mismo vehículo; el fragmento se evaluaba antes que el track
    # principal, así que la regla de continuación (al nacer el track) no podía verlo.
    UNKNOWN_HOLD_SEC = 3.0
    FRAGMENT_WINDOW_SEC = 4.0

    def _is_fragment(self, ev: ROIEvent, x: float) -> bool:
        for t, kind, y, xe in self._confirmed:
            if abs(ev.timestamp - t) <= self.FRAGMENT_WINDOW_SEC and abs(x - xe) < 0.35 and abs(ev.centroid_y - y) < 0.2:
                return True
        return False

    def collect_finished(self, now: Optional[float] = None) -> List[ROIEvent]:
        """
        Cierra los tracks no vistos por `track_lost_sec` (regla 2) y devuelve los
        eventos resultantes. Llamar en cada iteración del consumidor.
        """
        now = now if now is not None else time.monotonic()
        events: List[ROIEvent] = []
        # Liberar (o descartar) los DESCONOCIDA retenidos
        keep = []
        for ready_at, ev, x in self._held_unknown:
            if now < ready_at:
                keep.append((ready_at, ev, x))
            elif self._is_fragment(ev, x):
                self.total_continuations += 1
                log.info("Evento DESCONOCIDA descartado: fragmento de un evento confirmado.",
                         extra={"track_id": ev.track_id})
            else:
                events.append(ev)
        self._held_unknown = keep
        for tid in list(self._tracks):
            rec = self._tracks[tid]
            lost = now - rec.last_seen_ts > self.track_lost_sec
            if rec.pending and (lost or now - rec.pending[2] >= self.event_hold_sec):
                kind, reason, _ = rec.pending
                rec.pending = None
                events.append(self._emit(rec, kind, reason))
            if not lost:
                continue
            del self._tracks[tid]
            if rec.event_triggered:
                continue
            if self.solo_cruces:
                # ¿Lo continuó otro track (cambio de ID)? Si no, espera `union_sec` por si nace uno, y se archiva
                if self.union_sec <= 0 or not self._unir_a_activo(rec, now):
                    self._perdidos.append((now, rec))
                continue
            ev = self._evaluate_lost(rec)
            if ev is None:
                self.total_dropped += 1
            elif ev.event_type == GateEvent.DESCONOCIDA:
                self._held_unknown.append((now + self.UNKNOWN_HOLD_SEC, ev, rec.last.x))
            else:
                events.append(ev)
        # Regla estricta: los perdidos sin cruce que nadie continuó se archivan (no se avisan)
        if self._perdidos:
            vigentes = []
            for t_decl, rec in self._perdidos:
                if now - t_decl <= self.union_sec:
                    vigentes.append((t_decl, rec))
                    continue
                ev = self._evaluate_lost(rec)
                if ev is None:
                    self.total_dropped += 1
                else:
                    self.total_sin_cruce += 1
                    events.append(ev)
            self._perdidos = vigentes
        return events

    def cleanup_stale_tracks(self, max_age_sec: float = 15.0) -> int:
        """Compatibilidad: elimina tracks viejos sin evaluarlos."""
        now = time.monotonic()
        stale = [tid for tid, rec in self._tracks.items() if now - rec.last_seen_ts > max_age_sec]
        for tid in stale:
            del self._tracks[tid]
        return len(stale)

    def is_candidate(self, track_id: int) -> bool:
        """True si el track ya tocó PORTAL/INTERIOR (vale la pena guardar sus frames)."""
        rec = self._tracks.get(track_id)
        return rec is not None and rec.max_y >= self.line_b_ratio

    def draw_overlay(
        self, frame: np.ndarray, active_track_ids: Optional[List[int]] = None
    ) -> np.ndarray:
        """Dibuja las líneas ROI y contadores sobre el frame (cualquier escala)."""
        h, w = frame.shape[:2]
        line_a_y = int(h * self.line_a_ratio)
        line_b_y = int(h * self.line_b_ratio)

        cv2.line(frame, (0, line_a_y), (w, line_a_y), (0, 0, 0), 4)
        cv2.line(frame, (0, line_a_y), (w, line_a_y), (0, 100, 255), 2)
        cv2.putText(frame, "LINEA A (portal/rampa)", (10, line_a_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

        cv2.line(frame, (0, line_b_y), (w, line_b_y), (0, 0, 0), 4)
        cv2.line(frame, (0, line_b_y), (w, line_b_y), (0, 255, 100), 2)
        cv2.putText(frame, "LINEA B (calle/portal)", (10, line_b_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 2)

        stats_txt = f"IN: {self.total_entries}  OUT: {self.total_exits}  ?: {self.total_unknown}"
        cv2.rectangle(frame, (w - 260, 8), (w - 8, 40), (0, 0, 0), -1)
        cv2.putText(frame, stats_txt, (w - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        return frame

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    def relevant_track_count(self, near_margin: float = 0.08) -> int:
        """
        Tracks que justifican seguir corriendo YOLO sin movimiento: los que están cerca del portón
        o ya lo tocaron. Un auto estacionado en la vereda de enfrente (y≈0,38) mantenía YOLO
        trabajando todo el día aunque la escena estuviera quieta.
        """
        limit = self.line_b_ratio - near_margin
        return sum(1 for rec in self._tracks.values()
                   if rec.points and (any(p.zone != Zone.CALLE for p in rec.points)
                                      or (rec.last.y >= limit and self._is_moving(rec))))

    @staticmethod
    def _is_moving(rec: "TrackRecord", window_sec: float = 4.0, min_delta: float = 0.01) -> bool:
        # Un auto estacionado junto al portón (y≈0,48 en la grabación del 2026-10-02) no se mueve:
        # no justifica correr YOLO con la escena quieta.
        recent = [p for p in rec.points if rec.last.t - p.t <= window_sec]
        if len(recent) < 2:
            return True
        ys, xs = [p.y for p in recent], [p.x for p in recent]
        return max(ys) - min(ys) > min_delta or max(xs) - min(xs) > min_delta

    # ── Helpers privados ─────────────────────────────────────────────────────

    def _cruce(self, rec: TrackRecord) -> Optional[GateEvent]:
        """
        ¿El vehículo cruzó el portón? Normal: zona del primer punto → zona actual. Estricta: sobre toda la
        trayectoria (también la unida): estuvo en la CALLE y ahora está en el INTERIOR → ENTRADA; estuvo en el
        INTERIOR y ahora está en la CALLE → SALIDA. En ambos casos hace falta llegar a la línea del otro lado.
        """
        cur = rec.last.zone
        if not self.solo_cruces:
            first = rec.first.zone
            if first == Zone.CALLE and cur == Zone.INTERIOR:
                return GateEvent.ENTRADA
            if first == Zone.INTERIOR and cur == Zone.CALLE:
                return GateEvent.SALIDA
            return None
        antes = rec.points[:-1]
        if cur == Zone.INTERIOR and any(p.zone == Zone.CALLE for p in antes):
            return GateEvent.ENTRADA
        if cur == Zone.CALLE and any(p.zone == Zone.INTERIOR for p in antes):
            return GateEvent.SALIDA
        return None

    # Unión de trayectorias partidas (solo regla estricta): distancia máxima entre donde se perdió un track y
    # donde nace el siguiente (fracción del cuadro). Un auto que entra baja ~0,1 del cuadro por segundo.
    UNION_DX = 0.15
    UNION_DY = 0.15

    def _distancia_union(self, old: TrackRecord, nuevo: TrackRecord) -> Optional[float]:
        """
        ¿`nuevo` continúa a `old` (mismo vehículo con otro ID)? Nace después de la última vez que se vio a `old`,
        pegado a ese lugar, y siguiendo el mismo sentido: `old` estuvo en la CALLE y `nuevo` no nace en la CALLE
        (sigue entrando), o `old` estuvo en el INTERIOR y `nuevo` no nace en el INTERIOR (sigue saliendo).
        Devuelve la distancia (para elegir el más cercano) o None si no corresponde.
        """
        fin, born = old.last, nuevo.first
        if born.t < fin.t:
            return None
        dx, dy = abs(born.x - fin.x), abs(born.y - fin.y)
        if dx > self.UNION_DX or dy > self.UNION_DY:
            return None
        entra = born.zone != Zone.CALLE and any(p.zone == Zone.CALLE for p in old.points)
        sale = born.zone != Zone.INTERIOR and any(p.zone == Zone.INTERIOR for p in old.points)
        if not (entra or sale):
            return None
        return dx * dx + dy * dy

    def _unir(self, old: TrackRecord, rec: TrackRecord) -> None:
        rec.points = old.points + rec.points
        if len(rec.points) > 120:
            del rec.points[1:-100]
        self.total_unidos += 1
        log.info("Trayectoria unida: el seguimiento se había cortado a mitad del portón.",
                 extra={"track_id": rec.track_id, "track_anterior": old.track_id,
                        "y_corte": round(old.last.y, 3), "y_nace": round(rec.points[len(old.points)].y, 3)})

    def _unir_con_perdido(self, rec: TrackRecord) -> None:
        """Track recién nacido: ¿continúa a uno ya dado por perdido (sin cruzar) hace menos de `union_sec`?"""
        mejor, mejor_d = None, None
        for k, (t_decl, old) in enumerate(self._perdidos):
            if rec.first.t - t_decl > self.union_sec:
                continue
            d = self._distancia_union(old, rec)
            if d is not None and (mejor_d is None or d < mejor_d):
                mejor, mejor_d = k, d
        if mejor is not None:
            _, old = self._perdidos.pop(mejor)
            self._unir(old, rec)

    def _unir_a_activo(self, old: TrackRecord, now: float) -> bool:
        """
        Track que se acaba de dar por perdido sin cruzar: ¿lo continuó un track que nació mientras tanto (el
        tracker le dio otro ID antes de que este venciera)? Si sí, se le agrega su recorrido y se evalúa el cruce.
        """
        mejor, mejor_d = None, None
        for rec in self._tracks.values():
            if rec is old or rec.event_triggered or rec.pending:
                continue
            d = self._distancia_union(old, rec)
            if d is not None and (mejor_d is None or d < mejor_d):
                mejor, mejor_d = rec, d
        if mejor is None:
            return False
        self._unir(old, mejor)
        kind = self._cruce(mejor)
        if kind is not None:
            mejor.pending = (kind, "CRUCE_COMPLETO", now)
            self._last_event = (now, kind, mejor.last.y, mejor.last.x)
        return True

    def _is_continuation(self, rec: TrackRecord) -> bool:
        if self._last_event is None:
            return False
        t_ev, kind, y_ev, x_ev = self._last_event
        born = rec.first
        if not (-1.0 <= born.t - t_ev <= self.continuation_sec):
            return False
        if kind == GateEvent.ENTRADA:
            # sigue acercándose a la cámara, del mismo lado del cuadro
            return born.zone == Zone.INTERIOR and born.y >= y_ev - 0.02 and abs(born.x - x_ev) < 0.35
        if kind == GateEvent.SALIDA:
            # sigue alejándose por el portal (un auto que llega desde la CALLE es otro vehículo)
            return born.zone == Zone.PORTAL and born.y <= y_ev + 0.05
        return False

    def _evaluate_lost(self, rec: TrackRecord) -> Optional[ROIEvent]:
        if not rec.points or rec.max_y < self.line_b_ratio:
            return None  # nunca salió de la calle: tráfico o auto estacionado enfrente

        dy = rec.last.y - rec.first.y
        dx = rec.last.x - rec.first.x
        reached_interior = rec.max_y >= self.line_a_ratio

        # Paso lateral por la vereda sin entrar (p. ej. auto maniobrando en el cordón)
        if not reached_interior and abs(dx) > 2.5 * abs(dy) and abs(dx) > 0.15:
            return None

        # Regla estricta: nada de lo que se pierde sin cruzar se avisa; sale marcado para archivarse
        estricto = self.solo_cruces
        if dy > self.min_disp:
            return self._emit(rec, GateEvent.ENTRADA, "TRACK_PERDIDO_DESPLAZAMIENTO", estricto)
        if dy < -self.min_disp:
            return self._emit(rec, GateEvent.SALIDA, "TRACK_PERDIDO_DESPLAZAMIENTO", estricto)

        # Una sola detección rozando la línea B: casi siempre un auto de la calle junto al cordón
        if len(rec.points) == 1 and rec.first.y < self.line_b_ratio + 0.02:
            return None

        # Sin cruce y sin desplazamiento claro. Con solo_movimiento el evento sale MARCADO: el consumidor lo
        # envía solo si el lector local encontró una patente (un auto real parado en la puerta) y lo descarta
        # si no (un auto estacionado afuera visto a través de la reja: nunca tiene patente en sus fotos).
        quieto = self.solo_movimiento or estricto
        first_zone = rec.first.zone
        if first_zone == Zone.INTERIOR:
            return self._emit(rec, GateEvent.SALIDA, "TRACK_PERDIDO_ZONA_INICIAL", quieto)
        if first_zone == Zone.CALLE:
            return self._emit(rec, GateEvent.ENTRADA, "TRACK_PERDIDO_ZONA_INICIAL", quieto)
        return self._emit(rec, GateEvent.DESCONOCIDA, "TRACK_PERDIDO_SIN_DIRECCION", quieto)

    def _emit(self, rec: TrackRecord, kind: GateEvent, reason: str, sin_movimiento: bool = False) -> ROIEvent:
        rec.event_triggered = True
        if not sin_movimiento:
            # Un objeto quieto no es un paso confirmado: no debe servir para tratar al auto siguiente como
            # "continuación" ni como "fragmento" (ver _is_continuation y _is_fragment).
            self._last_event = (rec.last.t, kind, rec.last.y, rec.last.x)
            if kind != GateEvent.DESCONOCIDA:
                self._confirmed.append((rec.last.t, kind, rec.last.y, rec.last.x))
        if sin_movimiento:
            self.total_quietos += 1
        elif kind == GateEvent.ENTRADA:
            self.total_entries += 1
        elif kind == GateEvent.SALIDA:
            self.total_exits += 1
        else:
            self.total_unknown += 1
        (log.debug if sin_movimiento else log.info)(
            f"✅ Paso por portal: {kind.value}" if not sin_movimiento else f"Objeto sin movimiento claro ({kind.value}): pendiente de patente",
            extra={
                "track_id": rec.track_id, "reason": reason, "n_points": len(rec.points),
                "y_first": round(rec.first.y, 3), "y_last": round(rec.last.y, 3),
            },
        )
        return ROIEvent(
            track_id=rec.track_id,
            event_type=kind,
            timestamp=rec.last.t,
            centroid_y=rec.last.y,
            bbox=rec.last_bbox,
            reason=reason,
            n_points=len(rec.points),
            sin_movimiento=sin_movimiento,
            norm_box=rec.last_norm,
        )

    def _inside_polygon(self, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(self._polygon, (float(x), float(y)), False) >= 0

    @staticmethod
    def _parse_polygon(ratios_str: str) -> Optional[np.ndarray]:
        """Parsea "x1,y1,x2,y2,..." (ratios 0..1). El test se hace en coordenadas normalizadas."""
        if not ratios_str.strip():
            return None
        try:
            vals = [float(v) for v in ratios_str.split(",")]
            if len(vals) < 6 or len(vals) % 2 != 0:
                raise ValueError("se requieren al menos 3 puntos (x,y)")
            pts = np.array([(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)], dtype=np.float32)
            return pts.reshape((-1, 1, 2))
        except ValueError:
            log.warning("ROI_POLYGON_RATIOS inválido, ignorando.", extra={"val": ratios_str})
            return None
