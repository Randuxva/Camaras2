"""
Pipeline de calidad de imagen para el Agente ANPR Edge (2026).

Módulo responsable de:
  1. Diagnosticar la condición lumínica del frame (NORMAL / BACKLIGHT / LOW_LIGHT / OVEREXPOSED).
  2. Aplicar corrección adaptativa (CLAHE en canal L del espacio LAB + gamma).
  3. Calcular blur score (varianza Laplaciana normalizada).
  4. Detectar motion blur vs desenfoque óptico.
  5. Validar geometricamente el crop antes de subirlo.

ORDEN CRÍTICO del pipeline (no modificar sin justificación):
  1. Bilateral denoise  → elimina ruido ANTES de CLAHE
  2. BGR → LAB          → aislar luminancia sin afectar colores del auto
  3. Gamma + CLAHE en L → recuperar sombras/contraluz
  4. LAB → BGR          → reconstruir imagen
  5. Unsharp mask       → realzar bordes tipográficos de la patente
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)

# ── Constantes ───────────────────────────────────────────────────────────────

# Luminosidad media (canal L en LAB, rango 0-255)
_LUM_LOW_LIGHT    = 40.0
_LUM_OVEREXPOSED  = 200.0

# Ratio fondo-claro / zona-central para detectar contraluz
_BACKLIGHT_RATIO_THRESHOLD = 2.2
_BACKLIGHT_SHADOW_PCT      = 0.30
_BACKLIGHT_HIGHLIGHT_PCT   = 0.15

# Ratio de anisotropía Sobel para distinguir motion blur
_MOTION_BLUR_ANISOTROPY_THRESHOLD = 0.45

# Thresholds de blur (varianza Laplaciana sobre crop re-escalado a 640px)
BLUR_UNACCEPTABLE = 100.0   # Descartar
BLUR_ACCEPTABLE   = 150.0   # Mínimo para ANPR
BLUR_SHARP        = 250.0   # Óptimo

# Aspect ratio vehicular válido (width/height)
_ASPECT_MIN = 0.65
_ASPECT_MAX = 2.90

# Densidad mínima de bordes (Canny) en el crop (evita falsos positivos de asfalto)
_MIN_EDGE_DENSITY = 0.020


# ── Estructuras de datos ─────────────────────────────────────────────────────

@dataclass
class LightingAnalysis:
    condition: str          # "NORMAL" | "BACKLIGHT" | "LOW_LIGHT" | "OVEREXPOSED"
    mean_global: float
    mean_sky: float         # Tercio superior del frame
    mean_center: float      # Región central (zona del vehículo)
    backlight_ratio: float
    pct_shadows: float      # % píxeles < 40 en región central
    pct_highlights: float   # % píxeles > 220 en región superior


@dataclass
class BlurAnalysis:
    laplacian_var: float
    is_acceptable: bool
    is_sharp: bool
    is_motion_blur: bool
    anisotropy: float


@dataclass
class CropValidation:
    is_valid: bool
    reason: str
    width: int
    height: int
    aspect_ratio: float
    edge_density: float


# ── Funciones de análisis ────────────────────────────────────────────────────

def analyze_lighting(frame: np.ndarray) -> LightingAnalysis:
    """
    Clasifica la condición lumínica del frame completo.
    Trabaja en escala de grises para velocidad (~0.5ms en 720p).
    """
    if frame is None or frame.size == 0:
        return LightingAnalysis("NORMAL", 128, 128, 128, 1.0, 0.0, 0.0)

    gray = (
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(frame.shape) == 3
        else frame
    )
    h, w = gray.shape

    # Zonas de interés
    sky_region    = gray[0:int(h * 0.30), :]
    center_region = gray[int(h * 0.25): int(h * 0.75), int(w * 0.20): int(w * 0.80)]

    mean_global  = float(np.mean(gray))
    mean_sky     = float(np.mean(sky_region))
    mean_center  = float(np.mean(center_region)) if center_region.size > 0 else mean_global
    backlight_ratio = (mean_sky + 1.0) / (mean_center + 1.0)

    pct_shadows    = float(np.sum(center_region < 40))    / max(center_region.size, 1)
    pct_highlights = float(np.sum(sky_region > 220))      / max(sky_region.size, 1)

    # Clasificación
    if mean_global < _LUM_LOW_LIGHT:
        condition = "LOW_LIGHT"
    elif (
        backlight_ratio > _BACKLIGHT_RATIO_THRESHOLD
        and pct_shadows > _BACKLIGHT_SHADOW_PCT
        and pct_highlights > _BACKLIGHT_HIGHLIGHT_PCT
    ):
        condition = "BACKLIGHT"
    elif mean_global > _LUM_OVEREXPOSED:
        condition = "OVEREXPOSED"
    else:
        condition = "NORMAL"

    return LightingAnalysis(
        condition=condition,
        mean_global=mean_global,
        mean_sky=mean_sky,
        mean_center=mean_center,
        backlight_ratio=backlight_ratio,
        pct_shadows=pct_shadows,
        pct_highlights=pct_highlights,
    )


def compute_blur_score(image: np.ndarray, normalize_width: int = 640) -> BlurAnalysis:
    """
    Calcula la varianza del Laplaciano normalizada a un ancho de referencia.
    Los thresholds son válidos para imágenes de vehículos a 640px de ancho.

    Motion blur detection: ratio de anisotropía Sobel H vs V.
    Si la varianza es asimétrica (> 0.45), hay motion blur horizontal.
    """
    if image is None or image.size == 0:
        return BlurAnalysis(0.0, False, False, False, 0.0)

    h, w = image.shape[:2]
    if w == 0:
        return BlurAnalysis(0.0, False, False, False, 0.0)

    # Reescalar a ancho de referencia para normalizar la métrica
    scale = normalize_width / float(w)
    resized = cv2.resize(
        image,
        (normalize_width, max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Motion blur: comparar gradientes horizontales vs verticales
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    var_x = float(np.var(sobel_x))
    var_y = float(np.var(sobel_y))
    total = var_x + var_y + 1e-9
    anisotropy = abs(var_x - var_y) / total

    is_motion_blur = anisotropy > _MOTION_BLUR_ANISOTROPY_THRESHOLD and total < 600.0

    return BlurAnalysis(
        laplacian_var=lap_var,
        is_acceptable=lap_var >= BLUR_ACCEPTABLE,
        is_sharp=lap_var >= BLUR_SHARP,
        is_motion_blur=is_motion_blur,
        anisotropy=anisotropy,
    )


def validate_crop(crop: np.ndarray, min_w: int, min_h: int) -> CropValidation:
    """
    Valida geométricamente y por contenido un crop de vehículo.
    Comprueba: tamaño mínimo, aspect ratio vehicular y densidad de bordes.
    """
    if crop is None or crop.size == 0:
        return CropValidation(False, "Crop vacío", 0, 0, 0.0, 0.0)

    h, w = crop.shape[:2]

    if w < min_w or h < min_h:
        return CropValidation(
            False, f"Resolución insuficiente ({w}x{h})", w, h, 0.0, 0.0
        )

    aspect = w / float(h)
    if not (_ASPECT_MIN <= aspect <= _ASPECT_MAX):
        return CropValidation(
            False, f"Aspect ratio no vehicular ({aspect:.2f})", w, h, aspect, 0.0
        )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / max(edges.size, 1)

    if edge_density < _MIN_EDGE_DENSITY:
        return CropValidation(
            False, f"Contenido pobre - sin bordes ({edge_density:.4f})", w, h, aspect, edge_density
        )

    return CropValidation(True, "OK", w, h, aspect, edge_density)


# ── Enhancer principal ───────────────────────────────────────────────────────

class ANPRImageEnhancer:
    """
    Pipeline de mejora de imagen adaptativo para ANPR en tiempo real.
    Instanciar una vez, reutilizar en el loop de procesamiento.

    Benchmarks en CPU Intel Core i5-12th gen (720p frame):
      analyze_lighting:  ~0.5ms
      enhance:           ~4ms
      compute_blur:      ~2ms
    """

    def __init__(self) -> None:
        # CLAHE preinstanciados (evita overhead de recreación por frame)
        self._clahe_std  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._clahe_aggr = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))

        # LUT de gamma precalculada para valores comunes
        self._gamma_lut_cache: dict[float, np.ndarray] = {}

    def _gamma_lut(self, gamma: float) -> np.ndarray:
        """Genera o recupera la LUT de corrección gamma."""
        if gamma not in self._gamma_lut_cache:
            table = np.array(
                [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
                dtype=np.uint8
            )
            self._gamma_lut_cache[gamma] = table
        return self._gamma_lut_cache[gamma]

    def _apply_gamma(self, channel: np.ndarray, gamma: float) -> np.ndarray:
        """Aplica corrección gamma mediante LUT (< 0.1ms)."""
        return cv2.LUT(channel, self._gamma_lut(gamma))

    def _unsharp_mask(self, img: np.ndarray, sigma: float = 1.0, strength: float = 0.5) -> np.ndarray:
        """Realce de bordes sin halos (Unsharp Masking)."""
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        return cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)

    def enhance(
        self, frame: np.ndarray, lighting: Optional[LightingAnalysis] = None
    ) -> Tuple[np.ndarray, LightingAnalysis]:
        """
        Aplica el pipeline completo de mejora.

        Args:
            frame:    Imagen BGR de entrada (frame completo o crop).
            lighting: Análisis lumínico pre-calculado (si None, se calcula internamente).

        Returns:
            Tuple[imagen_mejorada, análisis_lumínico]
        """
        if frame is None or frame.size == 0:
            return frame, LightingAnalysis("NORMAL", 128, 128, 128, 1.0, 0.0, 0.0)

        if lighting is None:
            lighting = analyze_lighting(frame)

        # 1. Denoise bilateral (preserva bordes tipográficos, ~3ms)
        denoised = cv2.bilateralFilter(frame, d=5, sigmaColor=50, sigmaSpace=50)

        # 2. Convertir a LAB para trabajar solo el canal de luminancia
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        # 3. Corrección específica por condición
        cond = lighting.condition
        if cond == "BACKLIGHT":
            l_gamma   = self._apply_gamma(l_ch, gamma=0.55)
            l_enhanced = self._clahe_aggr.apply(l_gamma)
        elif cond == "LOW_LIGHT":
            l_gamma   = self._apply_gamma(l_ch, gamma=0.45)
            l_enhanced = self._clahe_aggr.apply(l_gamma)
        elif cond == "OVEREXPOSED":
            l_gamma   = self._apply_gamma(l_ch, gamma=1.6)
            l_enhanced = self._clahe_std.apply(l_gamma)
        else:  # NORMAL
            l_enhanced = self._clahe_std.apply(l_ch)

        # 4. Recombinar y volver a BGR
        enhanced_lab = cv2.merge((l_enhanced, a_ch, b_ch))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # 5. Unsharp mask suave para patente más legible
        result = self._unsharp_mask(enhanced_bgr, sigma=1.0, strength=0.50)

        return result, lighting
