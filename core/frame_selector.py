"""
Selector del mejor frame de un buffer por track ID.

Para cada vehículo tracked, se mantiene un buffer circular de N frames.
Cuando el ROI engine confirma un evento, se selecciona el frame con mayor
score de calidad para subirlo al webhook.

Score = Laplacian_var * penalización_motion_blur * penalización_luz * factor_tamaño

El factor de tamaño favorece el recorte más grande (vehículo más cerca = más
píxeles de patente), que es lo que más influye en la lectura OCR.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from core.image_quality import BlurAnalysis, LightingAnalysis, compute_blur_score


@dataclass
class FrameCandidate:
    """Un recorte candidato con sus métricas de calidad (no se guarda el frame completo)."""
    crop: np.ndarray                # Crop del vehículo
    bbox: tuple[int, int, int, int] # (x1, y1, x2, y2)
    track_id: int
    timestamp: float                # time.monotonic()
    blur: BlurAnalysis
    lighting: Optional[LightingAnalysis]
    score: float                    # Score ponderado para selección
    truncated: bool = False         # Vehículo cortado por el borde del cuadro
    rejilla: bool = False           # Se ve a través de la reja/malla del portón cerrado


# Reja o malla del portón: bordes por toda la imagen. En 500 recortes reales, el 3,8 % superó estos
# umbrales y los 12 que se revisaron a ojo eran rejas; ningún auto sin reja pasó de 0,25 / 0,90.
_REJILLA_DENSIDAD = 0.28
_REJILLA_COBERTURA = 0.93


_CANNY_LOCK = threading.Lock()


def _canny_seguro(gray: np.ndarray) -> np.ndarray:
    """
    cv2.Canny de OpenCV 5.0.0 corrompe la memoria (0xC0000374, el proceso muere) con bordes muy densos cuando corre
    con 1 hilo, que es como lo deja ultralytics al importarse (cv2.setNumThreads(1)). Medido: con 1 hilo se cayó en
    2 de 21 mallas de prueba; con 2 hilos o más, en ninguna. Se fuerzan 2 hilos mientras dura la llamada.
    El candado evita que dos llamadas se pisen al restaurar la configuración.
    """
    with _CANNY_LOCK:
        previos = cv2.getNumThreads()
        if previos >= 2:
            return cv2.Canny(gray, 60, 150)
        cv2.setNumThreads(2)
        try:
            return cv2.Canny(gray, 60, 150)
        finally:
            cv2.setNumThreads(previos)


def es_rejilla(crop: np.ndarray) -> bool:
    """¿El recorte está cubierto por la reja del portón? (densidad de bordes alta y repartida por todo el cuadro)"""
    if crop is None or crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = gray.shape[:2]
    if w < 32 or h < 32:
        return False
    gray = cv2.resize(gray, (192, max(16, int(192 * h / w))), interpolation=cv2.INTER_AREA)
    edges = _canny_seguro(gray)
    if float(np.mean(edges > 0)) <= _REJILLA_DENSIDAD:
        return False
    eh, ew = edges.shape
    bloques = edges[: eh // 8 * 8, : ew // 8 * 8].reshape(eh // 8, 8, ew // 8, 8).max(axis=(1, 3))
    return float(np.mean(bloques > 0)) > _REJILLA_COBERTURA


# Área a partir de la cual un recorte más grande ya no suma (≈1300x900 px). En la validación,
# un auto en el portón medía ~600 px de ancho y su patente ~65 px (< 100 px recomendados),
# así que conviene preferir el momento en que el vehículo está más cerca de la cámara.
_AREA_CAP = 1_200_000.0


def _compute_score(
    blur: BlurAnalysis,
    lighting: Optional[LightingAnalysis],
    crop_area: int = 0,
    truncated: bool = False,
    rejilla: bool = False,
) -> float:
    """Función de score ponderado para seleccionar el mejor frame."""
    score = blur.laplacian_var
    # La malla tiene bordes por todos lados: sin esto su "nitidez" le ganaba a un cuadro limpio
    if rejilla:
        score *= 0.1

    # Favorecer recortes grandes: más píxeles de patente
    if crop_area > 0:
        score *= min(1.0, crop_area / _AREA_CAP) ** 0.5

    # Vehículo cortado por el borde del cuadro: la patente puede quedar afuera
    if truncated:
        score *= 0.35

    # Penalización por motion blur (50%)
    if blur.is_motion_blur:
        score *= 0.50

    # Penalización por condición de luz adversa (20%)
    if lighting and lighting.condition in ("LOW_LIGHT", "OVEREXPOSED"):
        score *= 0.80

    return score


class FrameBuffer:
    """
    Buffer circular per-track para acumular candidatos de frames
    y seleccionar el de mayor calidad al momento de un evento.

    Thread-safety: se usa desde un único thread consumidor, no necesita lock.
    """

    def __init__(self, max_size: int = 6, min_blur: float = 100.0) -> None:
        self.max_size = max_size
        self.min_blur = min_blur
        # dict: track_id -> list de FrameCandidate (circular, más reciente al final)
        self._buffers: Dict[int, List[FrameCandidate]] = defaultdict(list)

    def add(
        self,
        track_id: int,
        frame: Optional[np.ndarray],
        crop: np.ndarray,
        bbox: tuple[int, int, int, int],
        lighting: Optional[LightingAnalysis] = None,
        frame_shape: Optional[tuple[int, int]] = None,
    ) -> None:
        """
        Agrega un candidato al buffer del track_id.
        Si el crop no supera el mínimo de blur, lo descarta silenciosamente.
        `frame_shape` (alto, ancho) permite penalizar vehículos cortados por el borde.
        """
        if crop is None or crop.size == 0:
            return

        blur = compute_blur_score(crop)
        if blur.laplacian_var < self.min_blur and self._buffers.get(track_id):
            return  # Borroso y ya hay candidatos: descartar
        # Si el track no tiene ningún candidato se acepta igual (mejor una foto borrosa
        # para revisión manual que perder el vehículo).

        truncated = False
        if frame_shape is not None:
            fh, fw = frame_shape
            x1, y1, x2, y2 = bbox
            edge = 0.01
            truncated = y2 >= fh * (1 - edge) or x1 <= fw * edge or x2 >= fw * (1 - edge)
        rejilla = es_rejilla(crop)
        score = _compute_score(blur, lighting, crop.shape[0] * crop.shape[1], truncated, rejilla)
        candidate = FrameCandidate(
            crop=crop.copy(),  # copia: el frame original se descarta para no retener 14 MB por candidato
            bbox=bbox,
            track_id=track_id,
            timestamp=time.monotonic(),
            blur=blur,
            lighting=lighting,
            score=score,
            truncated=truncated,
            rejilla=rejilla,
        )

        buf = self._buffers[track_id]
        buf.append(candidate)
        if len(buf) > self.max_size:
            # Descartar el de PEOR score (no el más antiguo): en una entrada la mejor foto
            # es la del portón, y los frames posteriores (auto cortado junto a la cámara) son peores.
            # Primero se descartan los cortados, luego los que están detrás de la reja, luego el de menor score
            worst = min(range(len(buf)), key=lambda i: (not buf[i].truncated, not buf[i].rejilla, buf[i].score))
            del buf[worst]  # por índice: el __eq__ del dataclass no compara arrays numpy

    def get_best(self, track_id: int) -> Optional[FrameCandidate]:
        """
        Devuelve el candidato con mayor score para el track_id dado.
        Retorna None si el buffer está vacío o el track no existe.
        """
        top = self.get_top(track_id, 1)
        return top[0] if top else None

    def get_top(self, track_id: int, k: int = 3) -> List[FrameCandidate]:
        """Los k mejores candidatos, del mejor al peor (para lectura multi-frame)."""
        buf = self._buffers.get(track_id, [])
        # Primero los vehículos completos (en la validación, un primer plano lateral cortado junto
        # a la cámara le ganaba por tamaño a la vista frontal en el portón, que es la que muestra la patente).
        # Cualquier cuadro sin reja le gana a uno con reja: la patente se lee mal a través de la malla.
        return sorted(buf, key=lambda c: (not c.truncated, not c.rejilla, c.score), reverse=True)[:k]

    def clear_track(self, track_id: int) -> None:
        """Libera la memoria del buffer de un track procesado."""
        self._buffers.pop(track_id, None)

    def clear_stale(self, max_age_sec: float = 15.0) -> None:
        """
        Limpieza periódica: elimina buffers de tracks que no recibieron
        candidatos nuevos en los últimos max_age_sec segundos.
        """
        now = time.monotonic()
        stale = [
            tid for tid, buf in self._buffers.items()
            if buf and (now - max(c.timestamp for c in buf)) > max_age_sec
        ]
        for tid in stale:
            self._buffers.pop(tid, None)

    @property
    def active_tracks(self) -> int:
        return len(self._buffers)
