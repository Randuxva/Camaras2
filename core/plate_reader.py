"""
Lectura local de patentes en el portón (2026): detector de patente + OCR + fusión multi-cuadro.

POR QUÉ (prueba real 2026-10-01, docs/PRUEBA_CAMARA_2026-10-01.md):
  Se enviaba a Gemini el recorte del auto entero, comprimido y con CLAHE; el primer recorte era el
  más cercano, donde la patente sale cortada. Resultado: 3 de 13 patentes leídas, aunque ~8 eran
  legibles a ojo. Práctica estándar de ANPR: localizar la patente, leerla en varios cuadros y
  votar carácter por carácter; enviar a la nube la patente ampliada, no el auto.

  - Detector: open-image-models (YOLOv9-t end2end, ONNX, CPU).
  - OCR: fast-plate-ocr (CCT, ONNX), con probabilidad por carácter.
  - Fusión: voto ponderado por carácter entre las lecturas con formato argentino válido.

Si las librerías no están instaladas el agente sigue funcionando como antes (sin lectura local).
"""
from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)

# Mismos formatos que supabase/functions/process-anpr/logic.ts (PLATE_FORMATS)
PLATE_FORMATS = (
    ("AUTO_MERCOSUR", re.compile(r"^[A-Z]{2}\d{3}[A-Z]{2}$")),   # AA123BB (2016+)
    ("AUTO_1995", re.compile(r"^[A-Z]{3}\d{3}$")),               # ABC123
    ("MOTO_MERCOSUR", re.compile(r"^[A-Z]\d{3}[A-Z]{3}$")),     # A123BCD
    ("MOTO_2010", re.compile(r"^\d{3}[A-Z]{3}$")),              # 123ABC
)


def normalize_plate(text: Optional[str]) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def plate_format(plate: str) -> Optional[str]:
    for name, rx in PLATE_FORMATS:
        if rx.match(plate):
            return name
    return None


@dataclass
class PlateRead:
    """Lectura de la patente en un recorte de vehículo."""
    text: str                    # normalizada (sin espacios ni '_' de relleno)
    char_probs: List[float]
    det_conf: float
    box: Tuple[int, int, int, int]  # dentro del recorte del vehículo
    width_px: int
    sharpness: float             # varianza del Laplaciano sobre la patente
    plate_img: np.ndarray = field(repr=False)  # patente recortada (BGR, resolución original)
    source_index: int = -1       # índice del recorte de origen

    @property
    def format(self) -> Optional[str]:
        return plate_format(self.text)

    @property
    def min_prob(self) -> float:
        return float(min(self.char_probs)) if self.char_probs else 0.0

    @property
    def mean_prob(self) -> float:
        return float(np.mean(self.char_probs)) if self.char_probs else 0.0

    @property
    def quality(self) -> float:
        """Qué tan buena es esta vista de la patente (para ordenar los recortes que se envían)."""
        size = min(self.width_px / 140.0, 1.0)          # 100–150 px de ancho es lo recomendado
        sharp = min(self.sharpness / 300.0, 1.0)
        ocr = self.mean_prob if self.format else self.mean_prob * 0.5
        return round(self.det_conf * (0.35 + 0.65 * size) * (0.5 + 0.5 * sharp) * (0.3 + 0.7 * ocr), 4)


@dataclass
class FusedPlate:
    """Resultado de combinar las lecturas de varios cuadros del mismo vehículo."""
    plate: Optional[str]
    format: Optional[str]
    char_conf: List[float]       # acuerdo ponderado por posición (0..1)
    support: int                 # lecturas que coinciden exactamente con `plate`
    n_reads: int                 # lecturas con formato válido consideradas
    reads: List[str]

    @property
    def min_char_conf(self) -> float:
        return float(min(self.char_conf)) if self.char_conf else 0.0

    def is_strong(self, min_support: int = 2, min_char_conf: float = 0.75) -> bool:
        """Lectura local confiable por sí sola: formato válido, varios cuadros y sin dudas por carácter."""
        return bool(self.plate and self.format and self.support >= min_support
                    and self.min_char_conf >= min_char_conf)

    def to_payload(self) -> dict:
        return {
            "plate": self.plate,
            "format": self.format,
            "support": self.support,
            "n_reads": self.n_reads,
            "min_char_conf": round(self.min_char_conf, 3),
            "char_conf": [round(c, 3) for c in self.char_conf],
            "reads": self.reads,
            "strong": self.is_strong(),
        }


def fuse_reads(reads: Sequence[PlateRead]) -> FusedPlate:
    """
    Voto por carácter entre lecturas de formato válido y la misma longitud (la más votada).
    Cada lectura pesa su probabilidad por carácter: un carácter dudoso en un cuadro borroso
    no le gana a uno nítido en otro cuadro.
    """
    all_texts = [r.text for r in reads if r.text]
    valid = [r for r in reads if r.format]
    if not valid:
        return FusedPlate(None, None, [], 0, 0, all_texts)

    length = Counter(len(r.text) for r in valid).most_common(1)[0][0]
    same_len = [r for r in valid if len(r.text) == length]

    plate_chars, conf = [], []
    for i in range(length):
        votes: dict = defaultdict(float)
        for r in same_len:
            p = r.char_probs[i] if i < len(r.char_probs) else r.mean_prob
            votes[r.text[i]] += max(p, 1e-3)
        ch, w = max(votes.items(), key=lambda kv: kv[1])
        plate_chars.append(ch)
        conf.append(w / sum(votes.values()))

    plate = "".join(plate_chars)
    fmt = plate_format(plate)
    if not fmt:  # el voto mezcló formatos incompatibles: quedarse con la mejor lectura individual
        best = max(same_len, key=lambda r: r.mean_prob)
        plate, fmt = best.text, best.format
        conf = [c * 0.5 for c in best.char_probs]
    support = sum(1 for r in valid if r.text == plate)
    return FusedPlate(plate, fmt, conf, support, len(valid), all_texts)


def _sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class PlateReader:
    """Detector + OCR de patentes sobre recortes de vehículos (BGR, sin mejorar ni comprimir)."""

    def __init__(self, detector_model: str, ocr_model: str, min_det_conf: float = 0.35, threads: int = 0) -> None:
        from fast_plate_ocr import LicensePlateRecognizer
        from open_image_models.detection.factory import create_detector

        import onnxruntime as ort

        # La lectura corre en paralelo con YOLO: con todos los núcleos lo frenaba a 1–2 s por cuadro
        # en el Pentium de 2 núcleos. Automático: 1 hilo con ≤ 2 núcleos, la mitad con más (mini PC).
        if threads <= 0:
            cores = os.cpu_count() or 2
            threads = 1 if cores <= 2 else max(1, cores // 2)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.detector = create_detector(detector_model, conf_thresh=min_det_conf,
                                        providers=["CPUExecutionProvider"], sess_options=opts)
        self.ocr = LicensePlateRecognizer(ocr_model, device="cpu", sess_options=opts)
        self._color_mode = getattr(self.ocr.config, "image_color_mode", "grayscale")
        self._pad = getattr(self.ocr.config, "pad_char", "_")
        log.info(f"Lector local de patentes listo ({detector_model} + {ocr_model}).")

    @classmethod
    def try_create(cls, detector_model: str, ocr_model: str, threads: int = 0) -> Optional["PlateReader"]:
        try:
            return cls(detector_model, ocr_model, threads=threads)
        except Exception as e:  # sin librerías o sin modelos descargados: se sigue sin lectura local
            log.warning(f"Lectura local de patentes desactivada: {e}")
            return None

    def read(self, vehicle_crop: np.ndarray, source_index: int = -1) -> Optional[PlateRead]:
        """Detecta la patente más probable del recorte y la lee. None si no hay patente visible."""
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None
        dets = self.detector.predict(vehicle_crop)
        if not dets:
            return None
        d = max(dets, key=lambda x: x.confidence)
        b = d.bounding_box
        h, w = vehicle_crop.shape[:2]
        # margen chico: el OCR espera la patente completa, con borde
        mx, my = int((b.x2 - b.x1) * 0.06), int((b.y2 - b.y1) * 0.12)
        x1, y1, x2, y2 = max(0, b.x1 - mx), max(0, b.y1 - my), min(w, b.x2 + mx), min(h, b.y2 + my)
        plate_img = vehicle_crop[y1:y2, x1:x2]
        if plate_img.size == 0:
            return None

        if self._color_mode == "grayscale":
            ocr_in = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            ocr_in = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        pred = self.ocr.run_one(ocr_in, return_confidence=True)
        raw = pred.plate or ""
        probs = [float(p) for p in (pred.char_probs if pred.char_probs is not None else [])]
        # quitar relleno manteniendo alineadas las probabilidades
        kept = [(c, probs[i] if i < len(probs) else 0.0) for i, c in enumerate(raw) if c != self._pad and c.isalnum()]
        text = "".join(c for c, _ in kept).upper()
        return PlateRead(
            text=text,
            char_probs=[p for _, p in kept],
            det_conf=float(d.confidence),
            box=(x1, y1, x2, y2),
            width_px=int(b.x2 - b.x1),
            sharpness=_sharpness(plate_img),
            plate_img=plate_img,
            source_index=source_index,
        )

    def read_many(self, vehicle_crops: Sequence[np.ndarray]) -> List[PlateRead]:
        out = []
        for i, crop in enumerate(vehicle_crops):
            try:
                r = self.read(crop, source_index=i)
            except Exception as e:  # un recorte raro no debe tumbar el evento
                log.warning(f"Lectura local falló en un recorte: {e}")
                r = None
            if r is not None:
                out.append(r)
        return out


def upscale_plate(plate_img: np.ndarray, target_width: int = 480) -> np.ndarray:
    """Amplía la patente para el OCR en la nube (Gemini lee mejor caracteres grandes)."""
    h, w = plate_img.shape[:2]
    if w >= target_width or w == 0:
        return plate_img
    s = target_width / float(w)
    return cv2.resize(plate_img, (target_width, max(1, int(h * s))), interpolation=cv2.INTER_CUBIC)
