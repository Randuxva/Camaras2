"""
Color y tipo de vehículo SIN IA en la nube (respaldo cuando Gemini no responde o se queda sin cuota).

Solo usa OpenCV y NumPy (sin descargas ni modelos nuevos):
  * color_de(): mide la pintura en la zona del capó y las puertas, en el espacio Lab (la
    luminosidad separa negro de claro; el tono separa los colores con color).
  * tipo_de(): la clase del detector (auto, moto, camión, bus).

Es un respaldo: Gemini manda cuando contesta. Cada resultado trae una confianza (0 a 1) y la nube
solo lo usa si Gemini no dijo nada.

Medido con 116 vehículos cargados (tools/evaluar_color.py), sobre la foto ya realzada que se envía:
  * Familia correcta (OSCURO / CLARO / COLOR): 83 %.
  * Lo que la nube guarda como dato (confianza >= 0.5): NEGRO (acierta ~3 de 4). El ROJO ya no se guarda (auditoria del
    30/10/2026: 9 de 20 estaban mal, era la luz de freno) ni nada con luz baja, sobreexposicion o sin color.
  * GRIS / PLATA / BLANCO no se pueden separar de forma confiable (el realce aplana el brillo y
    aun una persona los confunde) y un azul/amarillo/verde detectado suele ser un auto gris con luz
    de color: se informa la familia, con confianza baja para que no se guarde como si fuera cierto.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# Zona de la pintura dentro del recorte del vehículo (capó y puertas; sin parabrisas, patente ni suelo)
_Y0, _Y1, _X0, _X1 = 0.48, 0.72, 0.28, 0.72

# Un píxel "tiene color" si su croma Lab supera esto; la zona "tiene color" si lo tiene una parte suficiente
_CROMA_PIXEL = 14.0
_FRACCION_COLOR = 0.30

# Luminosidad de la pintura (L de Lab, 0-100) en fotos ya realzadas por el agente
_L_NEGRO = 54.0
_L_BLANCO = 82.0
_L_PLATA = 71.0

# Tonos Lab (grados): rojo ≈ 40°, naranja ≈ 65°, amarillo ≈ 95°, verde ≈ 150°, azul ≈ 280°
_LIMITES_TONO = (28.0, 75.0, 115.0, 220.0, 330.0)

FAMILIA_DE = {"NEGRO": "OSCURO", "BLANCO": "CLARO", "GRIS": "CLARO", "PLATA": "CLARO", "BEIGE": "CLARO"}

_CLASE_A_TIPO = {"car": "AUTO", "motorcycle": "MOTO", "truck": "CAMIONETA", "bus": "OTRO"}


def familia_de(color: Optional[str]) -> str:
    return FAMILIA_DE.get(str(color or "").upper(), "COLOR") if color else "DESCONOCIDO"


@dataclass(frozen=True)
class ResultadoColor:
    color: Optional[str]
    confianza: float
    familia: str  # OSCURO | CLARO | COLOR | DESCONOCIDO

    def to_payload(self) -> dict:
        return {"colorLocal": self.color, "colorLocalConf": round(self.confianza, 2), "colorLocalFamilia": self.familia}


_NADA = ResultadoColor(None, 0.0, "DESCONOCIDO")


def _nombre_por_tono(tono: float, luz: float) -> str:
    if tono >= _LIMITES_TONO[4] or tono < _LIMITES_TONO[0]:
        return "ROJO"
    if tono < _LIMITES_TONO[1]:
        if tono < 45:
            return "ROJO"
        return "CAFE" if luz < 45 else "NARANJA"
    if tono < _LIMITES_TONO[2]:
        return "BEIGE" if luz >= 70 else "AMARILLO"
    if tono < _LIMITES_TONO[3]:
        return "VERDE"
    return "AZUL"


def color_de(crop: Optional[np.ndarray]) -> ResultadoColor:
    """Color de la pintura. `confianza` baja cuando el tono cae cerca de un límite o la luz es de gris/plata."""
    if crop is None or crop.size == 0 or crop.ndim != 3:
        return _NADA
    h, w = crop.shape[:2]
    if h < 40 or w < 40:
        return _NADA

    chico = cv2.resize(crop, (160, max(1, int(160 * h / w))), interpolation=cv2.INTER_AREA)
    lab_img = cv2.cvtColor(chico, cv2.COLOR_BGR2LAB).astype(np.float32)
    ch, cw = lab_img.shape[:2]
    zona = lab_img[int(ch * _Y0):int(ch * _Y1), int(cw * _X0):int(cw * _X1)].reshape(-1, 3)
    if zona.shape[0] < 50:
        return _NADA

    luz = float(np.median(zona[:, 0])) * 100.0 / 255.0
    a = zona[:, 1] - 128.0
    b = zona[:, 2] - 128.0
    con_color = np.hypot(a, b) > _CROMA_PIXEL
    fraccion = float(con_color.mean())

    if fraccion >= _FRACCION_COLOR:
        tono = math.degrees(math.atan2(float(np.median(b[con_color])), float(np.median(a[con_color])))) % 360.0
        nombre = _nombre_por_tono(tono, luz)
        cerca = min(min(abs(tono - l), 360.0 - abs(tono - l)) for l in _LIMITES_TONO)
        conf = min(1.0, 0.45 + fraccion * 0.4 + min(cerca, 20.0) / 80.0)
        # Solo el rojo se separa bien (7 de 9 en la medición). Un azul, amarillo o verde "detectado" suele ser un
        # auto gris u oscuro con luz de color: se informa, pero sin confianza para guardarlo como dato.
        return ResultadoColor(nombre, conf if nombre == "ROJO" else min(conf, 0.45), familia_de(nombre))

    if luz < _L_NEGRO:
        return ResultadoColor("NEGRO", min(1.0, 0.55 + (_L_NEGRO - luz) / 30.0), "OSCURO")
    if luz >= _L_BLANCO:
        return ResultadoColor("BLANCO", 0.4, "CLARO")
    # Entre gris y plata no hay frontera real: confianza baja a propósito (la nube no lo guarda como dato)
    return ResultadoColor("PLATA" if luz >= _L_PLATA else "GRIS", 0.3, "CLARO")


def tipo_de(clase_detector: Optional[str]) -> Optional[str]:
    """AUTO / MOTO / CAMIONETA / OTRO según la clase del detector; None si no hay clase."""
    return _CLASE_A_TIPO.get(str(clase_detector or "").lower().strip())


def atributos_de(crop: Optional[np.ndarray], clase_detector: Optional[str] = None) -> dict:
    """Lo que el agente manda a la nube como respaldo (se mezcla dentro de crop_meta)."""
    out = color_de(crop).to_payload()
    tipo = tipo_de(clase_detector)
    if tipo:
        out["tipoLocal"] = tipo
    return out
